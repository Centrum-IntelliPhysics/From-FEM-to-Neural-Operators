% From FEM to Neural Operators:
% Scientific machine learning for computational mechanics
% Bauhaus Spring School 2026

close all
clear 
clc

addpath('../FEM_lib');

%% ==================== PROBLEM SETUP =====================
a   = 0.0; b = 1.0;
Ne  = 10;                 % elements
p   = 1;                  % degree (1,2,3)
np  = p + 1;
ngp = max(p+1,2);         % robust rule for 1D

% Coefficients / sources
model.k_fun = @(x) 1 + 0*x;                % keep explicit for future generalization
model.f_fun = @(x) (pi^2)*sin(pi*x);       % body source

% (Optional) exact solution for error checks & plots
u_exact  = @(x) sin(pi*x);
du_exact = @(x) pi*cos(pi*x);

%% ==================== MESH & BOUNDARIES =================
mesh = mesh_1d_deg(a,b,Ne,p);
Ndof = numel(mesh.x);

% Dirichlet BCs (elimination)
mesh.bound.dirichlet = [1; Ndof];    % endpoints
mesh.bound.gD        = 0;            % scalar or vector matching the IDs

% (Optional) Neumann nodal loads
% mesh.bound.neumann = [];           
% mesh.bound.gN      = 0;            

%% ==================== PRECOMPUTE ([-1,1]) ==============
[gp,gw] = gauss_legendre(ngp);        % gp,gw on [-1,1]
[N,dN]  = basis1d_lagrange(p,gp);     % N,dN are (np×ngp)
pre = struct('p',p,'np',np,'ngp',numel(gp),'gp',gp,'gw',gw,'N',N,'dN',dN);

%% ==================== ASSEMBLY ==========================
[K,F] = assemble_system_1d(mesh, model, pre);

%% ==================== DIRICHLET (ELIMINATION) ==========
[K2,F2,uD,free] = apply_dirichlet(K,F,mesh);

%% ==================== SOLVE & RECONSTRUCT ==============
u = nan(Ndof,1);
u(free)       = K2 \ F2;
u(~isnan(uD)) = uD(~isnan(uD));

%% ==================== POST =============================
% Plots
[xplot, uplot, utrue] = fe_plot_samples_1d(mesh,u,p,u_exact, 20);
figure; plot(xplot, uplot,'-','LineWidth',1.5); hold on;
plot(xplot, utrue,'--','LineWidth',1.2); grid on; legend('FE','exact');
xlabel('x'); ylabel('u'); title(sprintf('Solution (p=%d)',p));

[xg, dudx_g, dudx_star] = fe_gradients_at_gauss_1d(mesh,u,pre,du_exact);
figure; plot(xg, dudx_g,'o','MarkerSize',4); hold on;
plot(xg, dudx_star,'.-','LineWidth',1.0); grid on;
legend('FE grad (Gauss pts)','exact grad'); xlabel('x'); ylabel('du/dx');

% Errors
errs = error_metrics_1d_gauss(mesh,u,model,pre,u_exact,du_exact);
fprintf('L2 error      = %.3e\n', errs.L2);
fprintf('Energy error  = %.3e\n', errs.energy);
