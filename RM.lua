--[[ Recurrent Memory Network
In this implementation, there is no padding for memory block
that is at the beginning of the sentence, the number of previous words fed into memory block
increasing from 1 to memory size (15 by default)
no waste of attention on those padding words

Author: Ke Tran <m.k.tran@uva.nl>
Date: 19/11/2015
--]]

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'xlua'

require 'util.misc'
local TextProcessor = require 'text.TextProcessor'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Recurrent Memory Network for Language Modeling')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir', 'data', 'data directory. Should contain train.txt, valid.txt and test.txt file')
-- model params
cmd:option('-rnn_size', 200, 'size of LSTM internal state')
cmd:option('-emb_size', 200, 'word embedding size')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-mem_size', 15, 'memory size')
cmd:option('-nhop', 1, 'number of hop')
cmd:option('-time', true, 'use temporal matrix in Memory Block')
cmd:option('-gate', true, 'use gating combination')
-- optimization
cmd:option('-learning_rate', 1, 'learning rate')
cmd:option('-decay_rate', 2, 'decay rate')
cmd:option('-learning_rate_decay_after',4,'in number of epochs, when to start decaying the learning rate')
cmd:option('-batch_size', 20, 'number of sentences to train in parallel')
cmd:option('-max_seq_length', 40, 'max number of timesteps to unroll during BPTT')
cmd:option('-min_seq_length', 15, 'min number of timesteps to unroll during BPTT')
cmd:option('-dropout', 0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-max_epochs', 20, 'number of full passes through the training data')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-max_grad_norm', 5, 'max norm of gradients')
-- bookkeeping
cmd:option('-seed',42,'torch manual random number generator seed')
cmd:option('-print_every',200,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','rm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-attfile', '', 'file storing attention weights for analysis')
cmd:option('-start_epoch', 1, 'start epoch when resuming training')
-- GPU/GPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local MemNN
if opt.gate then
    MemNN = require 'model.MemBlockG' -- experiment with a new Memory Block
    print('use gating combination')
else
    MemNN = require 'model.MemBlock'
end

if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = TextProcessor.create(opt.data_dir, opt.batch_size, opt.min_seq_length, opt.max_seq_length, 5, opt.mem_size)
local vocab_size = loader.vocab_size
local vocab = loader.vocab_mapping
-- need for visualization
local id2word = {}
for w,id in pairs(vocab) do
    id2word[id] = w
end

print('vocabulary size: ' .. vocab_size)

-- building model
-- shared LookupTable
local word_embeddings = nn.LookupTable(vocab_size, opt.emb_size)

-- output layer shared
mem_nn = MemNN.g_build_model({nwords=vocab_size, nhop=1, edim=opt.rnn_size, lindim=opt.rnn_size, mem_size=opt.mem_size})

-- create an LSTM
protos = {}
protos.rnn = LSTM.lstm(opt.emb_size, opt.rnn_size, opt.num_layers, opt.dropout)
protos.criterion = nn.ClassNLLCriterion()


-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone()) -- cell
end

-- output layer shared
local output_layer = nn.Sequential()
output_layer:add(nn.Linear(opt.rnn_size, vocab_size))
output_layer:add(nn.LogSoftMax())

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
    word_embeddings:cuda()
    output_layer:cuda()
    mem_nn:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn,  word_embeddings, mem_nn, output_layer)

-- initialization
do_random_init = true
if do_random_init then
    params:uniform(-0.05, 0.05) -- small uniform numbers
end

function init_forget_gate(rnn)
     for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
            end
        end
     end
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
init_forget_gate(protos.rnn)

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.max_seq_length, not proto.parameters)
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)

local time = torch.Tensor(opt.batch_size, opt.mem_size)
if opt.gpuid >= 0 then
    time = time:cuda()
end
-- fill in memory times
for t = 1, opt.mem_size do
    time:select(2, t):fill(t)
end


-- preprocessing helper function
function prepro(x,y)
    local c = {}
    local seq_length = x:size(2)
    for t = 1,seq_length do
        c[#c+1] = x:sub(1,-1,math.max(1,t-opt.mem_size+1),t):clone()
    end

    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y = y:transpose(1,2):contiguous()
    -- compute context for Memory Block
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        for _,ct in ipairs(c) do
            ct = ct:float():cuda()
        end
    end

    return x,y,c
end


function zero_time()
    for _,node in ipairs(mem_nn.forwardnodes) do
        if node.data.annotations.name == 'Ain_t' or node.data.annotations.name == 'Bin_t' then
            node.data.module.weight:zero()
        end
    end
end


function save_checkpoint(savefile, epoch)
    local checkpont = {}
    checkpont.params = params
    checkpont.learning_rate = opt.learning_rate
    checkpont.epoch = epoch
    torch.save(savefile, checkpont)
end

function train_minibatch()
    if not opt.time then
        zero_time()
    end

    grad_params:zero()
    ------------------- get minibatch ----------------
    local c
    local x, y  = loader:next_batch()
    x,y,c = prepro(x,y)

    local seq_length = x:size(1)
    
    ------------------- mixing forward and early backward -----------------
    local rnn_state = {[0] = init_state_global} -- bottom RNN

    local loss = 0
    -- forward massive word embeddings
    local embeddings = word_embeddings:forward(x)
    -- create grad back to word embeddings
    local grad_embs = torch.Tensor():typeAs(embeddings):resizeAs(embeddings):zero()
    local grad_in_rnn = {} -- gradient from output layer back into RNN

    for t = 1,seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{embeddings[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        local inp_mem_t = {lst[#lst]:clone(), c[t], time:sub(1,-1,math.max(opt.mem_size+1-t,1),-1):clone()}
        local mem_t = mem_nn:forward(inp_mem_t)
        -- early backprop trick
        local pred_t =  output_layer:forward(mem_t)
        loss = loss + clones.criterion[t]:forward(pred_t, y[t])
        local doutput_t = clones.criterion[t]:backward(pred_t, y[t])
        local drnn_t = output_layer:backward(mem_t, doutput_t)
        local grad_mem_t = mem_nn:backward(inp_mem_t, drnn_t)
        grad_in_rnn[t] = grad_mem_t[1]:clone() -- clone is needed here since the Memory block is shared
    end
    loss = loss / seq_length
    --- backward pass ---
    local drnn_state = {[seq_length] = clone_list(init_state, true)}
    local grad_in_mem
    for t=seq_length,1,-1 do
        -- backprop from loss to output
        local last = #drnn_state[t]
        drnn_state[t][last]:add(grad_in_rnn[t])
        local dlst = clones.rnn[t]:backward({embeddings[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {} -- gradient that comes to the previous state
        for k,v in pairs(dlst) do
            if k == 1 then
                grad_embs[t] = v
            else
                drnn_state[t-1][k-1]=v
            end
        end

    end
    word_embeddings:backward(x, grad_embs)
    local norm_dw = grad_params:norm()
    if norm_dw > opt.max_grad_norm then
        local shink_factor = opt.max_grad_norm/norm_dw
        grad_params:mul(shink_factor)
    end
    params:add(grad_params:mul(-opt.learning_rate))
    return loss
end


-- evaluate perplexity on the whole dataset
function eval(mode)
    if not opt.time then
        zero_time()
    end
    loader:load(mode)  -- load the correct dataset
    local loss = 0
    local rnn_state = {[0] = init_state}
    local nbatches = loader.nbatches
    local nwords = 0
    for i=1, nbatches do
        local x, y  = loader:next_batch()
        x, y, c = prepro(x,y,c)
        local seq_length = x:size(1)
        -- forward pass
        local embeddings = word_embeddings:forward(x)
        for t = 1,seq_length do
            clones.rnn[t]:evaluate()  -- for dropout to work properly
            local lst = clones.rnn[t]:forward{embeddings[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
            local inp_mem_t = {lst[#lst], c[t], time:sub(1,-1,math.max(opt.mem_size+1-t,1),-1):clone()}
            local mem_t = mem_nn:forward(inp_mem_t)
            -- go to output layer
            local pred_t =  output_layer:forward(mem_t)
            loss = loss + clones.criterion[t]:forward(pred_t, y[t])
        end
        nwords = nwords + seq_length
        if i % 10 == 0 then collectgarbage() end
    end
    loss = loss / nwords
    return loss
end


function tensor2str(t)
    local s = {}
    for i=1,t:nElement() do
        table.insert(s, string.format("%.4f", t[i]))
    end
    local x = stringx.join(' ', s)
    return x
end

function print_attention(mode, hop)
    if not opt.time then
        zero_time()
    end
    local h = hop or 1
    local file = io.open(opt.attfile, 'w')
    loader:load(mode)  -- load the correct dataset
    local loss = 0
    local rnn_state = {[0] = init_state}
    local nbatches = loader.nbatches
    local nwords = 0
    local grad_att_weights = torch.zeros(opt.batch_size, opt.mem_size)
    for i=1, nbatches do
        local x, y  = loader:next_batch()
        x, y, c = prepro(x,y,c)
        local seq_length = x:size(1)
        -- note that we don't need to go to the second lstm and to the output layer
        -- forward pass
        local embeddings = word_embeddings:forward(x)
        local atts = {}
        for t = 1,seq_length do
            clones.rnn[t]:evaluate()  -- for dropout to work properly
            local lst = clones.rnn[t]:forward{embeddings[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
            local inp_mem_t = {lst[#lst], c[t], time:sub(1,-1,math.max(opt.mem_size+1-t,1),-1):clone()}
            local mem_t = mem_nn:forward(inp_mem_t)
            local att_w
            -- go through the graph and extract attention weights
            for _,node in ipairs(mem_nn.forwardnodes) do
                if node.data.annotations.name == 'attention_' .. h then
                    att_w = node.data.module.output
                end
            end
            atts[t] = att_w:clone()
        end
        local sentences = {}
        for t = 1,seq_length-1 do -- ignoring </s> when printing out
            for j = 1,opt.batch_size do
                sentences[j] = sentences[j] or {}
                sentences[j][t] = string.format("%s\t%s", id2word[y[t][j]], tensor2str(atts[t][j]))
            end
        end
        for j = 1,opt.batch_size do
            local sent = sentences[j]
            for _,s in pairs(sent) do
                file:write(s .. '\n')
            end
            file:write('\n')
        end

        if i % 10 == 0 then collectgarbage() end
    end
    file:close()
end

-- being helpful since Das5 people are getting angry
local start_epoch = opt.start_epoch
-- start optimization here
function main()
    if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
    -- start training and evaluating

    local valid_ppl = torch.ones(opt.max_epochs):mul(math.huge)
    local test_ppl = torch.ones(opt.max_epochs):mul(math.huge)
    for epoch = start_epoch,opt.max_epochs do
        local train_loss = 0
        loader:load('train')
        nbatches = loader.nbatches
        for i = 1,nbatches do
            local loss = train_minibatch()
            train_loss = train_loss + loss
            if i % opt.print_every == 0 then
                print(string.format("epoch = %d, train perplexity = %6.8f\n", epoch, math.exp(train_loss/i)))
                xlua.progress(i, nbatches)
            end
            -- free up memory sometimes
            if i%10 == 0 then collectgarbage() end
        end

        local valid_loss = eval('valid')
        print(string.format("validation perplexity = %6.8f", math.exp(valid_loss)))

        -- decay learning rate
        if epoch > opt.learning_rate_decay_after then
            opt.learning_rate = opt.learning_rate/opt.decay_rate
        end

        print("Peeking into test perplexity!")
        local test_loss = eval('test')
        print(string.format("test perplexity = %6.8f", math.exp(test_loss)))
        print(string.format("current learning rate = %.10f", opt.learning_rate))

        valid_ppl[epoch] = math.exp(valid_loss)
        test_ppl[epoch] = math.exp(test_loss)
        best_valid_ppl, e = torch.min(valid_ppl, 1)

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, valid_loss)
        print('save model to ' .. savefile)
        save_checkpoint(savefile, epoch)
        print(string.format('===> Best validation perplexity = %6.8f\ttest perplexity = %6.8f', best_valid_ppl[1], test_ppl[e[1]]))
    end
end

if opt.init_from == '' then
    main()
else
    local checkpont = torch.load(opt.init_from)
    print('loading saved model from checkpont: ' .. opt.init_from)
    params:copy(checkpont.params)
    print('restart with learning rate: ' .. checkpont.learning_rate)
    opt.learning_rate = checkpont.learning_rate
    start_epoch = checkpont.epoch + 1
    if opt.attfile ~= '' then
        print_attention('valid')
    else
        main()
    end
end

