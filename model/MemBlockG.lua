local MemNN = {}
function MemNN.build_memory(params, input, context, time, noise)
    local hid = {}
    hid[0] = input
    local shareList = {}
    shareList[1] = {}

    local Ain_c = nn.LookupTable(params.nwords, params.edim)(context)
    local Ain_t = nn.LookupTable(params.mem_size, params.edim)(time):annotate{name='Ain_t'}
    local Ain = nn.CAddTable()({Ain_c, Ain_t})

    local Bin_c = nn.LookupTable(params.nwords, params.edim)(context)
    local Bin_t = nn.LookupTable(params.mem_size, params.edim)(time):annotate{name='Bin_t'}
    local Bin = nn.CAddTable()({Bin_c, Bin_t})
    -- utility
    function new_input_sum(xv,hv)
        local i2h = nn.Linear(params.edim, params.edim)(xv)
        local h2h = nn.Linear(params.edim, params.edim)(hv)
        return nn.CAddTable()({i2h, h2h})
    end

    for h=1, params.nhop do
        -- view as 3D Tensor
        -- this is necessary for attention
        local hid3dim = nn.View(1, -1):setNumInputDims(1)(hid[h-1])
        local MMaout = nn.MM(false, true)
        local Aout = MMaout({hid3dim, Ain})

        local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
        local P = nn.SoftMax()(Aout2dim):annotate{name='attention_' .. h}
        local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
        local MMbout = nn.MM(false, false)
        local Bout3dim = MMbout({probs3dim, Bin})
        local Bout = nn.View(-1):setNumInputDims(2)(Bout3dim)
        
        -- take the output of memory net and interpolate it with the input in a clever way
        -- first, use gates to decide which one we trust more
        -- we use a form of GRU: http://arxiv.org/abs/1412.3555
        local update_gate = nn.Sigmoid()(new_input_sum(Bout, hid[h-1])) -- compute z
        local reset_gate = nn.Sigmoid()(new_input_sum(Bout, hid[h-1])) -- compute r

        local gated_hidden = nn.CMulTable()({reset_gate, hid[h-1]}) -- element wise mult z and h-1

        local p2 = nn.Linear(params.edim, params.edim)(gated_hidden)
        local p1 = nn.Linear(params.edim, params.edim)(Bout)
        -- compute hidden candidate
        local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
        
        local zh = nn.CMulTable()({update_gate, hidden_candidate})
        local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), hid[h-1]})
        local next_h = nn.CAddTable()({zh, zhm1})
        table.insert(shareList[1], update_gate)
        table.insert(shareList[1], reset_gate)
        hid[h] = next_h
    end

    return hid, shareList
end

function MemNN.g_build_model(params)
    local input = nn.Identity()()
    local target = nn.Identity()()
    local context = nn.Identity()()
    local time = nn.Identity()()
    local hid, shareList = MemNN.build_memory(params, input, context, time)
    local model = nn.gModule({input, context, time}, {hid[#hid]})

    for i = 1,#shareList do
        local m1 = shareList[i][1].data.module
        for j = 2,#shareList[i] do
            local m2 = shareList[i][j].data.module
            m2:share(m1,'weight','bias','gradWeight','gradBias')
        end
    end
    return model
end

return MemNN
