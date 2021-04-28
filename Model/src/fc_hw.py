import numpy as np

def fc(ifm, ker, bias):

    cout_unit = 4 
    cin_blk_size = 64*39
    input_units = ifm.shape[-1]
    output_units = bias.shape[-1]
    tmp = np.zeros(output_units)
    
    for ci in range(0, input_units, cin_blk_size):
        for co in range(0, output_units, cout_unit):
            # fetch fm
            ifm_blk = ifm[ci : ci + cin_blk_size]
            # fetch weight
            ker_blk = ker[ci : ci + cin_blk_size, co : co + cout_unit]
            # compute
            ipa_out = ifm_blk.dot(ker_blk)
            # output control
            if ci == 0:
                ipa_out += bias[co : co + cout_unit]
            else:
                ipa_out += tmp[co : co + cout_unit]
            # write partial sum to output buffer
            # if ci == ceil(input_units / 64):
            #   cut ipa_out to 8 bit (truncate and round)
            # else:
            #   cut ipa_out to 16 bit (truncate)
            tmp[co : co + cout_unit] = ipa_out
    return tmp           
    

    
