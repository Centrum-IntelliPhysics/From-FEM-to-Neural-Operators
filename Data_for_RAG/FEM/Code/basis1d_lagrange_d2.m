function d2N = basis1d_lagrange_d2(p, xi)
% 1D Lagrange basis second derivatives on reference [-1,1].
%
% Inputs
%   p   : polynomial degree (1,2,3)
%   xi  : vector of evaluation points in [-1,1]
%
% Output
%   d2N : (np x ngp) second derivatives d^2N/dxi^2 at xi, with np = p+1

assert(isscalar(p) && p>=1 && p==floor(p) && p<=3, ...
    'p must be integer and take either value 1, 2 or 3');
xi = xi(:).';                 % row vector (1 x ngp)
ngp = numel(xi);
np  = p+1;

d2N = zeros(np, ngp);

switch p
    case 1
        d2N(:,:) = 0;

    case 2
        % nodes: -1, 0, +1
        d2N(1,:) =  1;
        d2N(2,:) = -2;
        d2N(3,:) =  1;

    case 3
        % nodes: -1, -1/3, +1/3, +1
        d2N(1,:) = (-54*xi + 18)/16;
        d2N(2,:) = ( 162*xi - 18)/16;
        d2N(3,:) = (-162*xi - 18)/16;
        d2N(4,:) = ( 54*xi + 18)/16;

    otherwise
        error('Only p = 1, 2, 3 are implemented.');
end
end
