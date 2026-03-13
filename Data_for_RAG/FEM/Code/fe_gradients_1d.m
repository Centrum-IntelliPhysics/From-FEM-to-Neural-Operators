function [xg, dudx_g, dudx_exact] = fe_gradients_1d(mesh,u,pre,du_exact)
% Gradients du/dx evaluated at Gauss points (1D).
% Using the isoparametric mapping:
%   u(ξ) = Σ_i N_i(ξ) u_i,  x(ξ) = Σ_i N_i(ξ) x_i
%   du/dx = (1/J) * du/dξ,  J = dx/dξ

Ne  = size(mesh.elem,1);
ngp = numel(pre.gp);

xg = zeros(Ne*ngp,1);
dudx_g = zeros(Ne*ngp,1);
dudx_exact = NaN(Ne*ngp,1);

useExact = (nargin >= 4) && ~isempty(du_exact);

for e = 1:Ne
    nodes = mesh.elem(e,:);
    coord = mesh.x(nodes);   % np×1
    ue    = u(nodes);        % np×1

    for q = 1:ngp
        idx = (e-1)*ngp + q;

        Nq  = pre.N(:,q);    % np×1
        dNq = pre.dN(:,q);   % np×1

        % Map Gauss point to physical coordinate
        xq = Nq.' * coord;          % scalar

        % Jacobian and inverse Jacobian
        Jq = dNq.' * coord;         % scalar
        invJq = 1/Jq;

        % Physical derivative of shapes, then du/dx at this Gauss point
        dNdx_q = invJq*dNq;       % np×1
        duq    = dNdx_q' * ue;     % scalar

        xg(idx)     = xq;
        dudx_g(idx) = duq;

        if useExact
            dudx_exact(idx) = du_exact(xq);
        end
    end
end
end
