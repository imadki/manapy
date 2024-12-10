from numba import cuda


def device_search_element(a:'int32[:]', target_value:'int32') -> 'int32':
    for i in range(a.shape[0]):
        if a[i] == target_value:
            return 1
    return 0


def kernel_assign(arr_out: 'float[:]', value : 'float'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, arr_out.shape[0], stride):
        arr_out[i] = value


def device_compute_upwind_flux(w_l:'float', w_r:'float', u_face:'float', v_face:'float', w_face:'float', 
                    normal:'float[:]', flux_w:'float[:]'):
  
    sol = 0.
    sign = u_face * normal[0] + v_face * normal[1] + w_face * normal[2]

    if sign >= 0:
        sol = w_l
    else:
        sol = w_r

    flux_w[0] = sign * sol