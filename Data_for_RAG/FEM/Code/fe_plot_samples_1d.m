
function [xplot, uplot, uexact] = fe_plot_samples_1d(mesh,u,p,u_exact,ns_per_elem)
% Using the isoparametric mapping x(ξ)=Σ N_i(ξ) x_i.

if nargin<5 || isempty(ns_per_elem) 
    ns_per_elem = 10; 
end

xi = linspace(-1,1,ns_per_elem);
[Ni,~] = basis1d_lagrange(p, xi);     % (np×ns)

xplot = [];
uplot = [];
uexact= [];

for e = 1:size(mesh.elem,1)
    nodes = mesh.elem(e,:);
    coord = mesh.x(nodes);            % np×1
    ue    = u(nodes);                 % np×1

    % Isoparametric mapping of sample points
    xs = Ni'*coord;            % (ns×1)

    % FE samples
    us = Ni'*ue;               % (ns×1)

    % concatenate
    if e==1
        xplot = xs; uplot = us;
        if nargin>=4 && ~isempty(u_exact) 
            uexact = u_exact(xs); 
        else 
            uexact = NaN(size(xs)); 
        end
    else
        xplot = [xplot; xs(2:end)];
        uplot = [uplot; us(2:end)];
        if nargin>=4 && ~isempty(u_exact)
            uexact = [uexact; u_exact(xs(2:end))];
        else
            uexact = [uexact; NaN(ns_per_elem-1,1)];
        end
    end
end
end
