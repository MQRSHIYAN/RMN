--[[ Modified from https://github.com/karpathy/char-rnn/
which was modified from https://github.com/oxford-cs-ml-2015/practical6
modifier: Ke Tran <m.k.tran@uva.nl>

Date: 29/11/2015

In this implementation, sentences of the same length are put into one bucket.
This helps to avoid doing padding and it's more NLPish.
Attempt to feed data stochastically.
--]]


local TextProcessor = {}
TextProcessor.__index = TextProcessor

function TextProcessor.create(data_dir, batch_size, min_seq_length, max_seq_length, min_count)
    local self = {}
    setmetatable(self, TextProcessor)
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'valid.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')

    local train_tensor_file = path.join(data_dir, 'train.t7')
    local valid_tensor_file = path.join(data_dir, 'valid.t7')
    local test_tensor_file = path.join(data_dir, 'test.t7')

    -- try to be helpful here
    self.train_tensor_file = train_tensor_file
    self.valid_tensor_file = valid_tensor_file
    self.test_tensor_file = test_tensor_file
    self.batch_size = batch_size
    self.min_seq_length = min_seq_length
    self.max_seq_length = max_seq_length

    local run_prepro = false
    if not (path.exists(train_tensor_file) or path.exists(vocab_file)) then
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    end

    if run_prepro then
        print('one-time setup')
        TextProcessor.text_to_tensor(train_file, vocab_file, train_tensor_file, min_count, min_seq_length, max_seq_length, batch_size)
        TextProcessor.text_to_tensor(valid_file, vocab_file, valid_tensor_file, min_count, min_seq_length, max_seq_length, batch_size)
        TextProcessor.text_to_tensor(test_file, vocab_file, test_tensor_file, min_count, min_seq_length, max_seq_length, batch_size)
    end

    print('load data files...')
    self.vocab_mapping = torch.load(vocab_file)
    self.vocab_size = 0
    for word, id in pairs(self.vocab_mapping) do
        self.vocab_size = self.vocab_size + 1
    end
    collectgarbage()
    return self
end

function TextProcessor:load(mode)
    local data
    if mode == 'train' then
        data = torch.load(self.train_tensor_file)
    elseif mode == 'valid' then
        data = torch.load(self.valid_tensor_file)
    elseif mode == 'test' then
        data = torch.load(self.test_tensor_file)
    else
        error('args: train, valid, or test')
    end

    -- organizing sentences into bucket
    self.nbatches = #data.xs
    if mode == 'train' then
        -- shuffling the data only during training
        local perm = torch.randperm(self.nbatches)

        self.x = {}
        self.y = {}
        for i = 1,self.nbatches do
            table.insert(self.x, data.xs[perm[i]])
            table.insert(self.y, data.ys[perm[i]])
        end
    else
        self.x = data.xs
        self.y = data.ys
    end
    print(string.format('data load done. Number of batches: %d', self.nbatches))
    collectgarbage()
    return self
end


function TextProcessor:next_batch()
    if self.nbatches > 1 then
        self.nbatches = self.nbatches - 1
    end
    local i = self.nbatches
    return self.x[i], self.y[i]
end

-- *** STATIC METHOD ***
function TextProcessor.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, min_count, min_seq_length, max_seq_length, batch_size)
    print('loading text file...')
    local tot_len = 0
    local vocab_mapping
    local unordered = {}
    -- creating buckets, key is sentence length and value is a tensor of all sentences with length = key
    local buckets = {}
    local tot_sent = 0
    local kept_sent = 0
    for line in io.lines(in_textfile) do
        local words = stringx.split(line)
        tot_sent = tot_sent + 1
        -- note that we add <s>
        local l = #words + 1
        if l <= max_seq_length and l >= min_seq_length then
            kept_sent = kept_sent + 1
            buckets[l] = (buckets[l] or 0) + 1
            for _,word in ipairs(words) do
                unordered[word] = (unordered[word] or 0) + 1
            end
        end
    end
    print(string.format('keep %d / %d', kept_sent, tot_sent))

    if not path.exists(out_vocabfile) then
        -- create vocabulary
        print('creating vocabulary mapping...')
        -- filter out low freq words and sort into a table
        local ordered = {}
        for word, count in pairs(unordered) do
            if count > min_count then
                ordered[#ordered+1] = word
            end
        end
        local vocab_size = #ordered
        table.sort(ordered)
        vocab_mapping = {}
        -- invert `ordered` to create word->int mapping
        for i, word in ipairs(ordered) do
            vocab_mapping[word] = i
        end

        local special_words = {"<s>", "</s>", "<unk>"}
        for i, word in pairs(special_words) do
            vocab_mapping[word] =  vocab_mapping[word]  or (vocab_size + i)
        end
        print('vocabulary size when created: ' .. vocab_size)
    else
        print('load vocabulary')
        vocab_mapping = torch.load(out_vocabfile)
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')

    local data = {}
    local curr_len = {}
    for l,n in pairs(buckets) do
        data[l] = torch.IntTensor(l*n)
        curr_len[l] = 0
    end

    for line in io.lines(in_textfile) do
        words = stringx.split(line)
        local l = #words + 1
        if l <= max_seq_length and l >= min_seq_length then
            local curr = curr_len[l]
            curr = curr + 1
            data[l][curr] = vocab_mapping["<s>"]  -- add beginning of sentence
            for _,word in pairs(words) do
                curr = curr + 1
                data[l][curr] = vocab_mapping[word] or vocab_mapping["<unk>"]
            end
            curr_len[l] = curr
        end
    end

    local xs, ys = {}, {}
    for l,x in pairs(data) do
        local len = x:size(1)
        local nb = math.floor(len/(batch_size*l))  -- number of batches
        local r = len % (batch_size*l)  -- the remain

        if nb > 0 then
            local buff_x = x:sub(1, batch_size*l * nb)
            local y = buff_x:clone()
            y:sub(1,-2):copy(buff_x:sub(2,-1))
            -- put to batches
            local tmpx = buff_x:view(batch_size, -1):split(l, 2) -- #rows = #batches
            local tmpy = y:view(batch_size, -1):split(l, 2) -- #rows = #batches
            assert(#tmpx == #tmpy)
            for i = 1,#tmpx do
                xs[#xs+1] = tmpx[i]
                ys[#ys+1] = tmpy[i]
                -- set end of setence properly
                ys[#ys]:sub(1,-1,-1,-1):fill(vocab_mapping["</s>"])
            end

        end
        --[[
        -- comment out at the moment,
        -- if use this, we need to reshape initialization of the hidden lstm
        if r > 0 and r/l > 2 then
            -- put all the remainder into one batch
            local buff_x = x:sub(nb*batch_size*l+1, len)
            local y = buff_x:clone()
            y:sub(1,-2):copy(buff_x:sub(2,-1))
            local rbs = r/l  -- remain batch size
            local tmpx = buff_x:view(rbs, -1):split(l, 2)
            local tmpy = y:view(rbs, -1):split(l,2 )
            tmpy[1]:sub(1,-1,-1,-1):fill(vocab_mapping["</s>"])
            xs[#xs+1] = tmpx[1]
            ys[#ys+1] = tmpy[1]
            print(tmpx[1]:size())
            print(tmpx[1]:size())
        end
        --]]
        
    end
    -- packing to data
    local savedata = { xs = xs, ys = ys}
    -- save output processed files
    if not path.exists(out_vocabfile) then
        print('saving ' .. out_vocabfile)
        torch.save(out_vocabfile, vocab_mapping)
    end
    print('saving ' .. out_tensorfile)

    torch.save(out_tensorfile, savedata)
end

return TextProcessor
