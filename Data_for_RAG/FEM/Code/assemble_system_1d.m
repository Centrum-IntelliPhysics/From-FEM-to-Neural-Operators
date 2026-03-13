
function [K,F] = assemble_system_1d(mesh, f_fun, pre)
% Assemble global K,F for 1D Poisson (unit coefficient).
%
% Lecture 2 mapping (isoparametric, parent element):
%   x(ξ) = Σ_i N_i(ξ) x_i
%   J(ξ) = dx/dξ = Σ_i dN_i/dξ x_i
%   dN/dx = (1/J) dN/dξ
%
% Lecture 3 algorithm:
%   - Loop over Gauss points q inside an element loop.
%   - Keep local (a,b) loops vectorized (outer products), i.e., no loops over a,b.

x    = mesh.x;
elem = mesh.elem;
Ndof = size(x,1);
Ne   = size(elem,1);
np   = pre.np;
ngp  = pre.ngp;

K = zeros(Ndof, Ndof);
F = zeros(Ndof,1);

for e = 1:Ne
    nodes = elem(e,:);            % 1×np
    coord = x(nodes);             % np×1 (local coordinates x_i)

    % ---- Isoparametric mapping quantities at Gauss points ----
    % (Computed vectorized once; Gauss integration done in q-loop below)
    xq = (pre.N.' * coord);       % 1×ngp
    J  = (pre.dN.' * coord);      % 1×ngp

    % Warning if element is inverted / degenerate at any Gauss point
    if any(J <= 0)
        warning('Element %d has non-positive Jacobian at some Gauss points (min J = %.3e).', ...
            e, min(J));
    end

    invJ = 1 ./ J;                % 1×ngp

    % Initialize element contributions
    Ke = zeros(np,np);
    Fe = zeros(np,1);

    % ---- Gauss-point loop ----
    for q = 1:ngp
        % Shape functions and derivatives at this Gauss point
        Nq    = pre.N(:,q);                 % np×1
        dNdxi = pre.dN(:,q);                % np×1

        % Physical derivative at this Gauss point: dN/dx = (1/J) dN/dξ
        dNdxq = invJ(q) * dNdxi';            % np×1

        % Source at this Gauss point (right hand side)
        fq = f_fun(xq(q));

        % Add contributions (vectorized over local indices via outer products)
        Ke = Ke + (dNdxq' * dNdxq) * J(q) * pre.gw(q);   % np×np
        Fe = Fe +  Nq * fq * J(q) * pre.gw(q);         % np×1
    end

    % ---- Scatter-add to global K and F ----
    K(nodes,nodes) = K(nodes,nodes) + Ke;
    F(nodes)       = F(nodes) + Fe;
end
end
