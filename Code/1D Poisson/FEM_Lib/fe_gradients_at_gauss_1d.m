function [xg, dudx_g, dudx_exact] = fe_gradients_at_gauss_1d(mesh,u,pre,du_exact)
%FE_GRADIENTS_AT_GAUSS_1D  Evaluate du/dx at Gauss points elementwise.
%
% Inputs
%   mesh     : .x (Ndof×1), .elem (Ne×np)
%   u        : (Ndof×1) FE solution
%   pre      : .gp, .gw, .N, .dN  (reference [-1,1])
%   du_exact : function handle (du*/dx)(x) (optional, for overlay)
%
% Outputs
%   xg         : concatenated Gauss point coordinates (column)
%   dudx_g     : FE gradients at xg (column)
%   dudx_exact : exact gradients at xg (NaN if not provided)

Ne = size(mesh.elem,1);
ngp = numel(pre.gp);

xg = zeros(Ne*ngp,1);
dudx_g = xg;
dudx_exact = NaN(size(xg));

k = 1;
for e = 1:Ne
    nodes = mesh.elem(e,:);
    coord = mesh.x(nodes);              % np×1
    xa = coord(1); xb = coord(end);
    h  = xb - xa; xc = 0.5*(xa+xb);

    xs   = (xc + (h/2)*pre.gp).';       % (ngp×1)
    ue   = u(nodes);                    % np×1
    dNdx = (2/h)*pre.dN;                % np×ngp
    dus  = (ue.' * dNdx).';             % (ngp×1)

    idx = k:(k+ngp-1);
    xg(idx)     = xs;
    dudx_g(idx) = dus;
    if nargin>=4 && ~isempty(du_exact)
        dudx_exact(idx) = du_exact(xs);
    end
    k = k + ngp;
end
end
