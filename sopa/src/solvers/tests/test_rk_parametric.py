import torch

from sopa.src.solvers.rk_parametric_order4stage4 import RKOrder4Stage4, build_ButcherTableau_RKStandard, build_ButcherTableau_RK38
from sopa.src.solvers.rk_parametric_order3stage3 import RKOrder3Stage3
from sopa.src.solvers.rk_parametric_order2stage2 import RKOrder2Stage2, build_ButcherTableau_Midpoint, build_ButcherTableau_Heun

def test_ButcherTableau_validity(c, w, b, order = 4):
    '''Check if coefficients in Butcher tableau satisfy requirements (1.9), (1.11)
    from chapter II.1 of "Hairer, Wanner. Solving ordinary differential equations I".
    Solvers of order p should satisfy
    (1.9) and (1.11 a-b) if p = 2,
    (1.9) and (1.11 a-d) if p = 3,
    (1.9) and (1.11 a-h) if p = 4.
    '''
    dtype = b[0].dtype
    device = b[0].device
    
    if dtype == torch.float64:
        eps = torch.finfo(torch.float32).eps 
    else:
        eps = torch.finfo(torch.float16).eps 
    
    # (1.9)
    for i in range(len(c)):
        assert (torch.abs(c[i] - sum(w[i])) < eps)
        
    # (1.11 a)
    assert (torch.abs(sum(b) - 1.) < eps)
    
    # (1.11 b)
    if order >= 2:
        assert (torch.abs(sum([bi * ci for bi, ci in zip(b, c)]) - 1/2.) < eps)

    # (1.11 c-d)
    if order >=3:
        assert (torch.abs(sum([bi * ci**2 for bi, ci in zip(b, c)]) - 1/3.) < eps)
        assert (torch.sum(sum([b[i] * w[i][j] * c[j] for i in range(order) for j in range(order) if j < i]) - 1/6.) < eps)
        
    # (1.11 e-h)
    if order >= 4:
        assert (torch.abs(sum([bi * ci**3 for bi, ci in zip(b, c)]) - 1/4.) < eps)
        assert (torch.abs(sum([b[i] * c[i] * w[i][j] * c[j] for i in range(order) for j in range(order) if j < i]) - 1/8.) < eps)
        assert (torch.abs(sum([b[i] * w[i][j] * c[j]**2 for i in range(order) for j in range(order) if j < i]) - 1/12.) < eps)
        assert (torch.abs(sum([b[i] * w[i][j] * w[j][k] * c[k] for i in range(order) for j in range(order) for k in range(order) if (j < i and k < j)]) - 1/24.) < eps)
    
    if order >= 5:
        raise NotImplementedError
        
        
def test_rk_parametric_validity(dtype = None, device = None):
    # test RKOrder2Stage2 solver
    for _ in range(100):
        u0 = torch.rand(1, dtype = dtype)
        rk_solver = RKOrder2Stage2('u', u0 = u0, dtype = dtype, device = device)
        rk_solver.freeze_params()

        c, w, b = rk_solver.build_ButcherTableau(return_tableau = True)
        test_ButcherTableau_validity(c, w, b, order = rk_solver.order)

    
    # test RKOrder3Stage3 solver
    for _ in range(100):
        u0, v0 = [torch.rand(1, dtype = dtype) for _ in range(2)]
        rk_solver = RKOrder3Stage3('uv', u0 = u0, v0 = v0, dtype = dtype, device = device)
        rk_solver.freeze_params()

        c, w, b = rk_solver.build_ButcherTableau(return_tableau = True)
        test_ButcherTableau_validity(c, w, b, order = rk_solver.order)

    
    # test RKOrder4Stage4 solver
    for parameterization in ['u1', 'u2', 'u3', 'uv'][:]:
        for _ in range(100):
            u0, v0 = [torch.rand(1, dtype = dtype) for _ in range(2)]
            rk_solver = RKOrder4Stage4(parameterization=parameterization, u0 = u0, v0 = v0, dtype = dtype, device = device)
            rk_solver.freeze_params()

            c, w, b = rk_solver.build_ButcherTableau(return_tableau = True)
            test_ButcherTableau_validity(c, w, b, order = rk_solver.order)
        
def test_ButcherTableau_equality(c, w, b, true_c, true_w, true_b):
    ''' Check whether two different ButcherTableau have similar coefficients
    '''
    dtype = b[0].dtype
    
    if dtype == torch.float64:
        eps = torch.finfo(torch.float32).eps 
    else:
        eps = torch.finfo(torch.float32).eps * 10
    
    assert torch.max(torch.abs(c - true_c)) <  eps

    for i, (w_i, true_w_i) in enumerate(zip(w, true_w)):
        assert torch.max(torch.abs(w_i - true_w_i)) < eps

    assert torch.max(torch.abs(b - true_b)) <  eps
        
        
def test_rk_parametric_order4stage4(dtype = None, device = None):
    ''' Check whether computed parameters of RKParametricSolver coincide with
    manually defined ones.
    '''
    # Check if when using (parameterization='u2', u0 = 1/3., v0 = 2/3.) 
    # coefficients coinside with standard RK method

    rk_solver = RKOrder4Stage4(parameterization='u2', u0=1/3., v0=2/3., dtype = dtype, device = device)
    rk_solver.freeze_params()

    c, w, b = rk_solver.build_ButcherTableau(return_tableau = True)
    true_c, true_w, true_b = build_ButcherTableau_RKStandard(dtype = dtype, device = device)

    test_ButcherTableau_validity(c, w, b, order = rk_solver.order)
    test_ButcherTableau_validity(true_c, true_w, true_b, order = 4)

    test_ButcherTableau_equality(*(c, w, b), *(true_c, true_w, true_b))
    
    
    # Check if when using (parameterization='uv', u0 = 1/3., v0 = 2/3.) 
    # coefficients coinside with standard RK 3/8 method

    rk_solver = RKOrder4Stage4(parameterization='uv', u0=1/3., v0=2/3., dtype = dtype, device = device)
    rk_solver.freeze_params()    

    c, w, b = rk_solver.build_ButcherTableau(return_tableau = True)
    true_c, true_w, true_b = build_ButcherTableau_RK38(dtype = dtype, device = device)
    
    test_ButcherTableau_validity(c, w, b, order = rk_solver.order)
    test_ButcherTableau_validity(true_c, true_w, true_b, order = 4)

    test_ButcherTableau_equality(*(c, w, b), *(true_c, true_w, true_b))
    
    
def test_rk_parametric_order2stage2(dtype = None, device = None):
    ''' Check whether computed parameters of RKParametricSolver coincide with
    manually defined ones.
    '''
    # Check if when using (u0 = 1/2.) 
    # coefficients coinside with Midpoint method

    rk_solver = RKOrder2Stage2('u', u0=1/2., dtype = dtype, device = device)
    rk_solver.freeze_params()

    c, w, b = rk_solver.build_ButcherTableau(return_tableau = True)
    true_c, true_w, true_b = build_ButcherTableau_Midpoint(dtype = dtype, device = device)

    test_ButcherTableau_validity(c, w, b, order = rk_solver.order)
    test_ButcherTableau_validity(true_c, true_w, true_b, order = 2)

    test_ButcherTableau_equality(*(c, w, b), *(true_c, true_w, true_b))
    
    
    # Check if when using (u0 = 1.) 
    # coefficients coinside with Heun's method

    rk_solver = RKOrder2Stage2('u', u0=1., dtype = dtype, device = device)
    rk_solver.freeze_params()    

    c, w, b = rk_solver.build_ButcherTableau(return_tableau = True)
    true_c, true_w, true_b = build_ButcherTableau_Heun(dtype = dtype, device = device)
    
    test_ButcherTableau_validity(c, w, b, order = rk_solver.order)
    test_ButcherTableau_validity(true_c, true_w, true_b, order = 2)

    test_ButcherTableau_equality(*(c, w, b), *(true_c, true_w, true_b))