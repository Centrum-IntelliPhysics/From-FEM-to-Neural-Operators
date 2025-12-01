function F2 = apply_neumann(F,mesh)
%APPLY_NEUMANN  Add nodal Neumann loads (point loads) at given node IDs (1D).
%
% Inputs
%   F    : (Ndof×1) load vector
%   mesh : struct with (optional)
%          - mesh.bound.neumann : vector of node IDs
%          - mesh.bound.gN      : scalar OR vector of values matching those IDs
%
% Output
%   F2   : updated load vector
%
% Notes
%   - This helper applies nodal values directly.
%   - For true boundary integrals in higher-D, you’d assemble edge/face terms.

F2 = F;

if ~isfield(mesh,'bound') || ~isfield(mesh.bound,'neumann') || isempty(mesh.bound.neumann)
    return;
end

Ndof = numel(F2);
ids = mesh.bound.neumann(:);
assert(all(ids>=1 & ids<=Ndof), 'Some Neumann node IDs are out of range.');

gN = 0;                             % default: zero if not provided
if isfield(mesh.bound,'gN'), gN = mesh.bound.gN; end

if isscalar(gN)
    F2(ids) = F2(ids) + gN;
else
    gN = gN(:);
    assert(numel(gN)==numel(ids), ...
        'Length of mesh.bound.gN must match number of Neumann nodes.');
    F2(ids) = F2(ids) + gN;
end
end
