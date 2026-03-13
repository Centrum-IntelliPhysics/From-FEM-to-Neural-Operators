function errs = error_metrics_1d(mesh,u,pre,u_exact,du_exact)
% L2 and energy errors via Gauss integration (1D Poisson).
%
% Finite element fields evaluated at Gauss points using the isoparametric mapping:
%   u(ξ) = Σ_i N_i(ξ) u_i,   x(ξ) = Σ_i N_i(ξ) x_i
%   du/dx = (1/J) * du/dξ,   J = dx/dξ = Σ_i x_i dN_i/dξ
%
% Error measures:
%   L2 error:
%     ||e||_{L2}^2 = ∫_Ω (u_h(x) - u(x))^2 dx
%   Energy error for unit diffusion (k=1):
%     ||e||_E^2 = ∫_Ω ( (du_h/dx) - (du/dx) )^2 dx
%
% Inputs:
%   mesh.x, mesh.elem  : 1D mesh coordinates and connectivity
%   u                  : nodal FE solution
%   pre.N, pre.dN      : shape functions and ξ-derivatives at Gauss points (np×ngp)
%   pre.gw             : Gauss weights (1×ngp or ngp×1)
%   u_exact, du_exact  : function handles, u_exact(x), du_exact(x)
%
% Output:
%   errs.L2, errs.energy

x    = mesh.x;
elem = mesh.elem;
Ne   = size(elem,1);
ngp  = numel(pre.gp);

% Checking if the exact solutions for u and du/dx are provided
have_u  = (nargin >= 4) && ~isempty(u_exact);
have_du = (nargin >= 5) && ~isempty(du_exact);

% Initializing the errors
L2sq = 0;
Esq  = 0;  

for e = 1:Ne
    nodes = elem(e,:);
    coord = x(nodes);      % np×1
    ue    = u(nodes);      % np×1

    for q = 1:ngp
        % Shape functions at this Gauss point
        Nq  = pre.N(:,q);     % np×1
        dNq = pre.dN(:,q);    % np×1

        % Map Gauss point to physical coordinate and compute Jacobian
        xq = Nq.'  * coord;   % scalar
        Jq = dNq.' * coord;   % scalar

        % FE solution and derivative at Gauss point
        uhq  = Nq'*ue;             % u_h(xq)
        duhq = (1/Jq)*dNq'*ue;     % du_h/dx at xq

        % Exact values
        if have_u
            uq = u_exact(xq);
        else
            uq = 0;
        end

        if have_du
            duq = du_exact(xq);
        else
            duq = 0;
        end

        % Accumulate squared errors
        L2sq = L2sq + (uhq - uq)^2 * Jq * pre.gw(q);
        Esq  = Esq  + (duhq - duq)^2 * Jq * pre.gw(q) ; 
    end
end

errs.L2     = sqrt(L2sq);
errs.energy = sqrt(Esq);
end
