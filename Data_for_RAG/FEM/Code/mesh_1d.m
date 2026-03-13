function mesh = mesh_1d(a,b,Ne,p)
% Build a 1D uniform mesh for degree-p finite elements on [a,b]
%
% Inputs
%   a, b : scalars with a < b (domain endpoints)
%   Ne   : positive integer, number of elements
%   p    : positive integer (1,2,3), polynomial degree
%
% Outputs
%   mesh.dim  = 1
%   mesh.x    = (Ndof x 1) node coordinates, Ndof = Ne*p + 1
%   mesh.elem = (Ne x (p+1)) connectivity
%   mesh.bound.dirichlet = []  (fill in the main script)
%   mesh.bound.neumann   = []  (fill in the main script)
%
% Notes
%   - Node numbering is 1..Ndof.
%   - Each element e uses nodes ((e-1)*p + 1) : ((e-1)*p + p + 1).

assert(isfinite(a) && isfinite(b) && a < b, 'Require a<b');
assert(isscalar(Ne) && Ne>=1 && Ne==floor(Ne), 'Ne must be a positive integer');
assert(isscalar(p) && p>=1 && p==floor(p) && p<=3, 'p must be integer and take either value 1, 2 or 3');

Ndof = Ne*p + 1;
x = linspace(a,b, Ndof).';               % (Ndof x 1)

elem = zeros(Ne, p+1);
for e = 1:Ne
    elem(e,:) = (e-1)*p + (1:(p+1));     % consecutive nodes
end

mesh.dim  = 1;
mesh.x    = x;
mesh.elem = elem;
mesh.bound.dirichlet = [];               % set in main script
mesh.bound.neumann   = [];               % set in main script
end
