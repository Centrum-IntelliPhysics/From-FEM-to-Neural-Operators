
function [N, dN] = basis1d_lagrange(p, xi)
% 1D Lagrange basis & derivatives on reference [-1,1].
%
% Inputs
%   p   : polynomial degree (1,2,3)
%   xi  : vector of evaluation points in [-1,1]
%
% Outputs
%   N   : (np x ngp) shape function values at xi, with np = p+1
%   dN  : (np x ngp) derivatives dN/dxi at xi
%
% Notes
%   - Use mapping x(ξ) = N(ξ)*x to go to physical space.
%   - Then dN/dx = (2/h) * dN/dxi for affine 1D elements.
%   - p=3 uses parent nodes {-1, -1/3, +1/3, +1}.

assert(isscalar(p) && p>=1 && p==floor(p) && p<=3, 'p must be integer and take either value 1, 2 or 3');
xi = xi(:).';                 % row vector (1 x ngp)
ngp = numel(xi);
np  = p+1;

N  = zeros(np, ngp);
dN = zeros(np, ngp);

switch p
    case 1
        % nodes: -1, +1
        N(1,:)  = 0.5*(1 - xi);
        N(2,:)  = 0.5*(1 + xi);
        dN(1,:) = -0.5;
        dN(2,:) =  0.5;

    case 2
        % nodes: -1, 0, +1
        N(1,:)  = 0.5*xi.*(xi - 1);
        N(2,:)  = 1 - xi.^2;
        N(3,:)  = 0.5*xi.*(xi + 1);
        dN(1,:) = xi - 0.5;
        dN(2,:) = -2*xi;
        dN(3,:) = xi + 0.5;

    case 3
        % nodes: -1, -1/3, +1/3, +1
        N(1,:)  = -9/16*(xi + 1/3).*(xi - 1/3).*(xi - 1);
        N(2,:)  =  27/16*(xi + 1   ).*(xi - 1/3).*(xi - 1);
        N(3,:)  = -27/16*(xi + 1   ).*(xi + 1/3).*(xi - 1);
        N(4,:)  =   9/16*(xi + 1   ).*(xi + 1/3).*(xi - 1/3);
        dN(1,:) = (-27*xi.^2 + 18*xi + 1)/16;
        dN(2,:) = ( 81*xi.^2 - 18*xi - 27)/16;
        dN(3,:) = (-81*xi.^2 - 18*xi + 27)/16;
        dN(4,:) = ( 27*xi.^2 + 18*xi - 1)/16;

    otherwise
        error('Only p = 1, 2, 3 are implemented.');
end
end
