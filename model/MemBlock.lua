local MemNN = {}
paths.dofile('LinearNB.lua')
function MemNN.build_memory(params, input, context, time)
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
        local Bout = MMbout({probs3dim, Bin})
        local C = nn.LinearNB(params.edim, params.edim)(hid[h-1])
        table.insert(shareList[1], C)
        local D = nn.CAddTable()({C, Bout})
        if params.lindim == params.edim then
            hid[h] = D
        elseif params.lindim == 0 then
            hid[h] = nn.ReLU()(D)
        else
            local F = nn.Narrow(2,1,params.lindim)(D)
            local G = nn.Narrow(2,1+params.lindim,params.edim-params.lindim)(D)
            local K = nn.ReLU()(G)
            hid[h] = nn.JoinTable(2)({F,K})
        end
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
