function [gp,gw] = gauss_legendre(n)
%GAUSS_LEGENDRE  Gauss–Legendre quadrature (points & weights) on [-1,1].
%
% Input
%   n   : number of points (supported: 1..4)
%
% Outputs
%   gp  : 1 x n points in [-1,1]
%   gw  : 1 x n weights for [-1,1]
%
% Example
%   [gp,gw] = gauss_legendre(3);   % gp=[-0.7746 0 0.7746], gw=[0.5556 0.8889 0.5556]
%
% Notes
%   - In 1D FEM with degree p, a robust default is n = max(p+1,2).
%   - Mapping to physical element [xa,xb]:
%         x(ξ) = N(ξ)*x,   wJ = (h/2) * gw

assert(isscalar(n) && n>=1 && n==floor(n), 'n must be a positive integer');
if n > 4
    error('gauss_legendre: only up to 4 points are implemented (n <= 4).');
end

switch n
    case 1
        gp = 0;
        gw = 2;
    case 2
        gp = [-0.577350269189626  0.577350269189626];
        gw = [ 1.000000000000000  1.000000000000000];
    case 3
        gp = [-0.774596669241483  0  0.774596669241483];
        gw = [ 0.555555555555556  0.888888888888889  0.555555555555556];
    case 4
        gp = [-0.861136311594053 -0.339981043584856  0.339981043584856  0.861136311594053];
        gw = [ 0.347854845137454  0.652145154862546  0.652145154862546  0.347854845137454];
end
end
