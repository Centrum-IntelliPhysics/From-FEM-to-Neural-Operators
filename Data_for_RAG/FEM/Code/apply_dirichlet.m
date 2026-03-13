
function [K2,F2,uD,free] = apply_dirichlet(K,F,mesh)
% Strong (elimination) imposition of Dirichlet boundary values.
%
% Inputs
%   K    : (Ndof×Ndof) global stiffness matrix
%   F    : (Ndof×1)    global load vector
%   mesh : struct with
%          - mesh.bound.dirichlet : vector of node IDs to prescribe
%          - mesh.bound.gD        : scalar OR vector of values (same length as IDs)
%
% Outputs
%   K2   : reduced stiffness for free DOFs (size = nFree × nFree)
%   F2   : reduced load     for free DOFs (size = nFree × 1)
%   uD   : (Ndof×1) prescribed values at Dirichlet nodes
%   free : column vector of free DOF indices such that K2*u(free) = F2
%

Ndof = size(K,1);
uD   = nan(Ndof,1);

% --- Validate & collect Dirichlet nodes
assert(isfield(mesh,'bound') && isfield(mesh.bound,'dirichlet'), ...
    'mesh.bound.dirichlet (node IDs) is required.');
ids = mesh.bound.dirichlet(:);
assert(~isempty(ids), 'Dirichlet node list is empty.');
assert(all(ids>=1 & ids<=Ndof), 'Some Dirichlet node IDs are out of range.');

% make IDs unique but preserve order (MATLAB R2019b+: "stable")
ids = unique(ids,'stable');

% --- Values (scalar or vector)
gD = 0;  % default homogeneous
if isfield(mesh.bound,'gD') 
    gD = mesh.bound.gD; 
end

if isscalar(gD)
    uD(ids) = gD;
else
    gD = gD(:);
    assert(numel(gD)==numel(ids), ...
        'Length of mesh.bound.gD must match number of Dirichlet nodes.');
    uD(ids) = gD;
end

% --- Split DOFs
dir  = find(~isnan(uD));
free = setdiff((1:Ndof).', dir);  % column vector

% --- Elimination: K_ff u_f = F_f - K_fd u_d
F2 = F - K(:,dir) * uD(dir);
K2 = K(free,free);
F2 = F2(free);
end
