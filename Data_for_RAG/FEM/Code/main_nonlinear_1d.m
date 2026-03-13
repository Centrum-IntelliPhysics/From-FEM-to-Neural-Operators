% From FEM to Neural Operators: Scientific machine learning for computational mechanics
% Bauhaus Spring School 2026
%
% main_nonlinear_1d.m
%
% Nonlinear 1D diffusion / Poisson-like problem:
%     - (k(u) u'(x))' = lambda * f(x)   in (a,b)
%     u = gD                           on Gamma_D (Dirichlet BC)
%     k(u) u' n = gN                   on Gamma_N (Neumann BC, optional)
%
% where k(u) is a nonlinear diffusion coefficient, e.g. k(u) = 1 + alpha*u^2.
%
% This script:
%   - sets up a 1D mesh and boundary conditions,
%   - defines the nonlinear material law k(u),
%   - performs load stepping in a load factor lambda in (0,1],
%   - calls a Newton solver for each lambda-step,
%   - reuses linear FEM infrastructure (mesh, basis, Gauss, Dirichlet/Neumann).
%
% Required functions (provided in linear FEM code):
%   mesh_1d, basis1d_lagrange, gauss_legendre,
%   apply_dirichlet, apply_neumann,
%   fe_plot_samples_1d, fe_gradients_1d (optional),
%   (error_metrics_1d only if you provide an exact solution).
%
% New functions for nonlinear FEM:
%   newton_nonlinear_1d
%   assemble_nonlinear_1d
%   element_res_tan_nonlinear_1d

close all
clear
clc

%% ==================== PROBLEM SETUP =====================
a   = 0.0; 
b   = 1.0;       % Domain from x=a to x=b
Ne  = 10;        % Number of elements
p   = 1;         % Polynomial degree (1,2,3)
np  = p + 1;     % Number of nodes per element
ngp = max(p+1,2);% Robust rule for integration

% Body load for the PDE: -(k(u) u')' = lambda * f(x)
% (Choose any smooth function; no exact solution is needed here.)
f_fun = @(x) (pi^2) * sin(pi*x);

% Nonlinear diffusion coefficient k(u) = 1 + alpha * u^2
alpha  = 100.0;                    % strength of nonlinearity
k_fun  = @(u) 1 + alpha * u.^2;  % nonlinear conductivity
dk_fun = @(u) 2 * alpha * u.^1;     % derivative dk/du

%% ==================== MESH & BOUNDARIES =================
mesh = mesh_1d(a,b,Ne,p);
Ndof = numel(mesh.x);

% Dirichlet BCs: endpoints fixed (here homogeneous: u=0 at both ends)
mesh.bound.dirichlet = [1; Ndof];    % node IDs
mesh.bound.gD        = 0;            % scalar or vector matching IDs

% Optional Neumann nodal loads (1D endpoint); leave empty for now
% mesh.bound.neumann = [];
% mesh.bound.gN      = 0;

%% ==================== PRECOMPUTE ([-1,1]) ==============
[gp,gw] = gauss_legendre(ngp);        % Gauss points/weights on [-1,1]
[N,dN]  = basis1d_lagrange(p,gp);     % N,dN are (np×ngp), dN = dN/dxi
pre = struct('p',p,'np',np,'ngp',numel(gp), ...
             'gp',gp,'gw',gw,'N',N,'dN',dN);

%% ==================== LOAD STEPPING PARAMETERS ==========
nSteps     = 50;                        % number of load steps
lambda_vec = linspace(1/nSteps, 1.0, nSteps);  % lambda in (0,1]

% Newton solver options
opts.maxIter = 20;       % maximum Newton iterations per load step
opts.tolRes  = 1e-8;     % residual tolerance (relative to initial)
opts.tolInc  = 1e-10;    % increment tolerance
opts.verbose = true;     % print iteration info

%% ==================== INITIAL GUESS =====================
% Start from zero (or you could use a linear solution as initial guess)
u = zeros(Ndof,1);

% Optional: enforce Dirichlet values in initial guess (here gD=0 anyway)
if isfield(mesh,'bound') && isfield(mesh.bound,'dirichlet')
    ids = mesh.bound.dirichlet(:);
    if isfield(mesh.bound,'gD')
        gD = mesh.bound.gD;
        if isscalar(gD)
            u(ids) = gD;
        else
            gD = gD(:);
            assert(numel(gD)==numel(ids), ...
                'Length of mesh.bound.gD must match number of Dirichlet nodes.');
            u(ids) = gD;
        end
    else
        u(ids) = 0;
    end
end

% Storage for solutions at each load step (useful for plotting)
u_hist = zeros(Ndof, nSteps);

%% ==================== LOAD STEP LOOP ====================
for s = 1:nSteps
    lambda = lambda_vec(s);
    fprintf('\n================ Load step %d / %d, lambda = %.3f ================\n', ...
        s, nSteps, lambda);
    
    % Scale the body load by lambda:
    f_scaled = @(x) lambda * f_fun(x);
    
    % Solve the nonlinear problem for this lambda using Newton
    [u, info] = newton_nonlinear_1d(mesh, u, pre, f_scaled, k_fun, dk_fun, opts);
    
    % Store solution
    u_hist(:,s) = u;
    
    % Report basic convergence info
    fprintf('  Newton converged in %d iterations. Final residual norm = %.3e\n', ...
        info.nIter, info.resNorm(end));
end

%% ==================== POSTPROCESSING ====================
% Plot final solution at full load (lambda ~ 1)
ns = 20;  % samples per element for plotting
[xplot, uplot] = fe_plot_samples_1d(mesh, u, p, [], ns);

figure;
plot(xplot, uplot, 'b-', 'LineWidth', 1.5); grid on;
xlabel('x'); ylabel('u(x)');
title(sprintf('Nonlinear FEM solution at full load (p=%d, \\alpha=%.1f)', p, alpha));

% % % % % Optional: plot the evolution with load stepping
% % % % figure; hold on; grid on;
% % % % colors = lines(nSteps);
% % % % for s = 1:nSteps
% % % %     [xs, us] = fe_plot_samples_1d(mesh, u_hist(:,s), p, [], ns);
% % % %     plot(xs, us, '-', 'Color', colors(s,:), ...
% % % %         'DisplayName', sprintf('\\lambda = %.2f', lambda_vec(s)));
% % % % end
% % % % xlabel('x'); ylabel('u(x)');
% % % % title('Nonlinear FEM solutions for different load levels \lambda');
% % % % legend('Location','best');





%% ===== Animated evolution with load stepping (single line) =====
figure;
hold on; grid on;

% Start with the first load step
[xs, us] = fe_plot_samples_1d(mesh, u_hist(:,1), p, [], ns);

% Create a single line object that we will update
hLine = plot(xs, us, 'b-', 'LineWidth', 1.5);

xlabel('x'); ylabel('u(x)');
title('Nonlinear FEM solution evolving with load factor \lambda');

% Add a small text label in normalized axes coordinates to show lambda
hTxt = text(0.05, 0.9, ...
    sprintf('\\lambda = %.2f', lambda_vec(1)), ...
    'Units','normalized', 'FontSize', 11, 'FontWeight', 'bold');

% Optionally fix y-limits to avoid rescaling during animation
% ylim_current = ylim;   % comment this and the next line out if you prefer autoscaling
% ylim(ylim_current);

xlim([0 1]);          % or [0 1] for your current setup
ylim([-0.1 1.2]);     % pick something comfortably large

for s = 1:nSteps
    % Sample FE solution at current load step
    [xs, us] = fe_plot_samples_1d(mesh, u_hist(:,s), p, [], ns);
    
    % Update line data
    set(hLine, 'XData', xs, 'YData', us);
    
    % Update lambda label
    set(hTxt, 'String', sprintf('\\lambda = %.2f', lambda_vec(s)));
    
    % Redraw figure
    drawnow;            % forces update immediately
    pause(0.2);         % slow down animation a bit (adjust or remove)
end