function [Re, KeT] = element_res_tan_nonlinear_1d(xe, ue, pre, f_fun, k_fun, dk_fun)
% element_res_tan_nonlinear_1d
%
% Compute the element residual vector and tangent matrix for a 1D
% nonlinear diffusion element:
%
%   - (k(u) u'(x))' = f(x)  on the element domain [x_e1, x_e2]
%
% Isoparametric mapping:
%   x(ξ)  = Σ_i N_i(ξ) x_i
%   J(ξ)  = dx/dξ = Σ_i dN_i/dξ x_i
%   dN_i/dx = (1/J) dN_i/dξ
%
% Local FE approximation:
%   u_h^{(e)}(x)  = Σ_b N_b^{(e)}(x) u_b^{(e)}
%   u_h^{(e)\prime}(x) = Σ_b u_b^{(e)} N_b^{(e)\prime}(x)
%
% Weak form contributions at local node a:
%   R_a^{(e)}(u^{(e)}) =
%      ∫_Ωe k(u_h^{(e)}(x)) u_h^{(e)\prime}(x) N_a^{(e)\prime}(x) dx
%      - ∫_Ωe f(x) N_a^{(e)}(x) dx
%
% Tangent matrix entries:
%   K_{T,ac}^{(e)} = ∂R_a^{(e)} / ∂u_c^{(e)}
%
% Inputs:
%   xe    : element node coordinates (np×1)
%   ue    : element nodal values (np×1)
%   pre   : struct with N, dN, gp, gw, np, ngp
%   f_fun : handle for body load f(x)
%   k_fun : handle for k(u)
%   dk_fun: handle for dk/du
%
% Outputs:
%   Re    : element residual vector (np×1)
%   KeT   : element tangent matrix (np×np)

np  = pre.np;
ngp = pre.ngp;

Re  = zeros(np,1);
KeT = zeros(np,np);

for q = 1:ngp
    % Shape functions and derivatives at Gauss point
    Nq   = pre.N(:,q);      % np×1
    dNq  = pre.dN(:,q);     % np×1 (dN/dxi)
    
    % Map Gauss point to physical coordinate and compute Jacobian
    xq = Nq.'  * xe;        % scalar x(ξ_q)
    Jq = dNq.' * xe;        % scalar dx/dξ at ξ_q
    
    if Jq <= 0
        warning('Warning: non-positive Jacobian J = %.3e at Gauss point %d.', Jq, q);
    end
    
    % Physical derivatives of shape functions: dN/dx = (1/J) dN/dxi
    dNdx = dNq / Jq;        % np×1
    
    % FE solution and derivative at Gauss point
    u    = Nq.'   * ue;     % u_h(xq)
    up   = dNdx.' * ue;     % du_h/dx at xq
    
    % Material and right-hand side
    k    = k_fun(u);        % k(u_h)
    dkdu = dk_fun(u);       % dk/du at u_h
    fq   = f_fun(xq);       % body load at xq
    
    % Gauss weight in physical coordinates
    wq = pre.gw(q) * Jq;
    
    %-------------------------------------------------------------
    % Residual contribution:
    %   R_a += [k(u)*u' N_a']_q * wq  -  [f N_a]_q * wq
    %
    % Vectorized form:
    %   Re += (k * up) * dNdx * wq  -  fq * Nq * wq
    %-------------------------------------------------------------
    Re = Re + (k * up) * dNdx * wq ...
           - fq * Nq * wq;
    
    %-------------------------------------------------------------
    % Tangent contribution:
    %
    %   ∂/∂u_c [k(u) u'] = k'(u) (∂u/∂u_c) u' + k(u) (∂u'/∂u_c)
    %                    = k'(u) N_c u' + k(u) N_c'
    %
    % so
    %   K_{T,ac} += [k N_c' + k'(u) u' N_c] N_a' * wq
    %
    % in vector form:
    %   col_c = k * dNdx + dkdu * up * Nq   (np×1)
    %   KeT  += dNdx * col_c' * wq          (np×np)
    %-------------------------------------------------------------
    col_c = k * dNdx + dkdu * up * Nq;  % np×1
    KeT   = KeT + dNdx * (col_c.') * wq;
end
end