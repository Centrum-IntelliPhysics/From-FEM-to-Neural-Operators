
% From FEM to Neural Operators: Scientific machine learning for computational mechanics
% Bauhaus Spring School 2026
%
% main script
%   Strong form (1D-Poisson equation):
%       -u''(x) = f(x)   in (a,b)
%       u = gD           on ΓD (Dirichlet, essential BC)
%       u' n = gN        on ΓN (Neumann, natural BC)
%
%   Weak form:
%       Find u in V such that for all v in V0,
%           a(u,v) = l(v)
%       with
%           a(u,v) = ∫_a^b u'(x) v'(x) dx
%           l(v)   = ∫_a^b f(x) v(x) dx  +  (Neumann boundary term, if present)

close all
clear
clc

%% ==================== PROBLEM SETUP =====================
a   = 0.0; b = 1.0;       % Domain from x=a to x=b
Ne  = 5;                 % number of elements
p   = 1;                  % polynomial degree (1,2,3)
np  = p + 1;              % number of nodes per element
ngp = max(p+1,2);         % robust rule for integration

% Poisson right-hand side: -u'' = f
f_fun = @(x) (pi^2)*sin(pi*x);   % body source

% (Optional) exact solution for error checks & plots
u_exact  = @(x) sin(pi*x);
du_exact = @(x) pi*cos(pi*x);

%% ==================== MESH & BOUNDARIES =================
mesh = mesh_1d(a,b,Ne,p);
Ndof = numel(mesh.x);

% Dirichlet BCs
mesh.bound.dirichlet = [1; Ndof];    % endpoints
mesh.bound.gD        = 0;            % scalar or vector matching IDs

% Optional Neumann nodal loads (1D endpoint)
% mesh.bound.neumann = [];           
% mesh.bound.gN      = 0;            % scalar or vector matching IDs

%% ==================== PRECOMPUTE ([-1,1]) ==============
[gp,gw] = gauss_legendre(ngp);        % Gauss points/weights on [-1,1]
[N,dN]  = basis1d_lagrange(p,gp);     % N,dN are (np×ngp), dN = dN/dxi
pre = struct('p',p,'np',np,'ngp',numel(gp),'gp',gp,'gw',gw,'N',N,'dN',dN);

%% ==================== ASSEMBLY ==========================
[K,F] = assemble_system_1d(mesh, f_fun, pre);

% Optional: add Neumann nodal loads (endpoint contribution in 1D)
F = apply_neumann(F,mesh);

%% ==================== DIRICHLET (ELIMINATION) ==========
[K2,F2,uD,free] = apply_dirichlet(K,F,mesh);

%% ==================== SOLVE & RECONSTRUCT ==============
u = nan(Ndof,1);
u(free)       = K2 \ F2;
u(~isnan(uD)) = uD(~isnan(uD));

%% ==================== POST =============================
% Plot solution
[xplot, uplot, utrue] = fe_plot_samples_1d(mesh,u,p,u_exact, 20);
figure; plot(xplot, uplot,'-','LineWidth',1.5); hold on;
plot(xplot, utrue,'--','LineWidth',1.2); grid on; legend('FE','exact');
xlabel('x'); ylabel('u'); title(sprintf('Solution (p=%d)',p));

% Plot gradients at Gauss points
[xg, dudx_g, dudx_star] = fe_gradients_1d(mesh,u,pre,du_exact);
figure; plot(xg, dudx_g,'o','MarkerSize',4); hold on;
plot(xg, dudx_star,'.-','LineWidth',1.0); grid on;
legend('FE grad (Gauss pts)','exact grad'); xlabel('x'); ylabel('du/dx');

% Errors:
errs = error_metrics_1d(mesh,u,pre,u_exact,du_exact);
fprintf('L2 error      = %.3e\n', errs.L2);
fprintf('Energy error  = %.3e\n', errs.energy);
