function [xplot, uplot, uexact] = fe_plot_samples_1d(mesh,u,p,u_exact,ns_per_elem)
%FE_PLOT_SAMPLES_1D  Sample u_h (and u*) smoothly along the mesh for plotting.
%
% Inputs
%   mesh        : .x (Ndof×1), .elem (Ne×np)
%   u           : (Ndof×1) FE solution
%   p           : polynomial degree (np = p+1)
%   u_exact     : function handle u*(x) (optional, for overlay)
%   ns_per_elem : integer #samples per element (default 10)
%
% Outputs
%   xplot  : concatenated coordinates across elements (column)
%   uplot  : FE values at xplot
%   uexact : exact values at xplot (NaN if u_exact not given)
%
% Notes
%   - Reference is [-1,1]. We sample xi in [-1,1], map to physical x.
%   - To avoid duplicate points at element interfaces, we skip the first
%     sample of each element except the very first.

if nargin<5 || isempty(ns_per_elem), ns_per_elem = 10; end

np = p+1;
[Ni,~] = basis1d_lagrange(p, linspace(-1,1,ns_per_elem));  % (np×ns)
xi = linspace(-1,1,ns_per_elem);

xplot = [];
uplot = [];
uexact= [];

for e = 1:size(mesh.elem,1)
    nodes = mesh.elem(e,:);
    coord = mesh.x(nodes);                     % np×1
    xa = coord(1); xb = coord(end);
    h  = xb - xa; xc = 0.5*(xa+xb);

    % physical samples
    xs = (xc + (h/2)*xi).';                    % (ns×1)

    % FE samples
    ue = u(nodes);                             % np×1
    us = (ue.' * Ni).';                        % (ns×1)

    % concatenate (avoid double-counting shared node)
    if e==1
        xplot = xs; uplot = us;
        if nargin>=4 && ~isempty(u_exact), uexact = u_exact(xs); else, uexact = NaN(size(xs)); end
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
