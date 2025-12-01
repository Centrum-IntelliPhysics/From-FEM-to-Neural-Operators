function errs = error_metrics_1d_gauss(mesh,u,model,pre,u_exact,du_exact)
%ERROR_METRICS_1D_GAUSS  L2 and energy errors via Gauss integration on [-1,1].
%
% Inputs
%   mesh   : .x (Ndof×1), .elem (Ne×np)
%   u      : (Ndof×1) FE solution vector
%   model  : struct with
%              .k_fun(x)  -> diffusion coefficient (scalar or vector); REQUIRED
%              (note: you decided to set model.k_fun = @(x) 1+0*x in main)
%   pre    : struct with reference quadrature/basis on [-1,1]
%              .gp (1×ngp), .gw (1×ngp), .N (np×ngp), .dN (np×ngp)
%   u_exact  : function handle u*(x)       (optional, for L2 error)
%   du_exact : function handle (du*/dx)(x) (optional, for energy error)
%
% Output
%   errs.L2        : sqrt( ∑_e ∑_q (u_h - u*)^2 * wJ )
%   errs.energy    : sqrt( ∑_e ∑_q k(xq) * (u'_h - u'_*)^2 * wJ )
%   errs.H1_semi   : sqrt( ∑_e ∑_q (u'_h - u'_*)^2 * wJ )   (unweighted)
%
% Notes
%   - If u_exact or du_exact are omitted, the corresponding term uses zero
%     (so you effectively get norms of u_h or u'_h).
%   - Gauss mapping: x = xc + (h/2)*gp, wJ = (h/2)*gw, dN/dx = (2/h) dN/dxi.

x    = mesh.x;
elem = mesh.elem;
Ne   = size(elem,1);

have_u  = (nargin>=5) && ~isempty(u_exact);
have_du = (nargin>=6) && ~isempty(du_exact);

L2sq = 0;  EEsq = 0;  H1sq = 0;

for e = 1:Ne
    nodes = elem(e,:);
    coord = x(nodes);                  % np×1
    xa = coord(1); xb = coord(end);
    h  = xb - xa;  xc = 0.5*(xa+xb);

    % Gauss in physical space
    xq = xc + (h/2)*pre.gp;            % 1×ngp
    wJ = (h/2)*pre.gw;                 % 1×ngp

    % FE values at Gauss points
    ue  = u(nodes);                    % np×1
    uhq = (ue.' * pre.N);              % 1×ngp
    duhq= (ue.' * ((2/h)*pre.dN));     % 1×ngp

    % Exact (optional)
    if have_u,  uq = u_exact(xq);  else, uq = 0*xq;  end
    if have_du, gq = du_exact(xq); else, gq = 0*xq;  end

    % Coefficient
    kq = model.k_fun(xq);              % 1×ngp

    % Accumulate
    L2sq = L2sq + sum( (uhq - uq).^2 .* wJ );
    H1sq = H1sq + sum( (duhq - gq).^2 .* wJ );
    EEsq = EEsq + sum( kq .* (duhq - gq).^2 .* wJ );
end

errs.L2       = sqrt(L2sq);
errs.H1_semi  = sqrt(H1sq);
errs.energy   = sqrt(EEsq);
end
