% Implementation of the Wirtinger Flow (WF) algorithm presented in the paper 
% "Phase Retrieval via Wirtinger Flow: Theory and Algorithms" 
% by E. J. Candes, X. Li, and M. Soltanolkotabi

% The input data are phaseless measurements about a random complex
% valued 1D signal. 

%% Make signal and data 
n = 128;
x = randn(n,1) + 1i*randn(n,1);

m = round(4.5*n);                     
A = 1/sqrt(2)*randn(m,n) + 1i/sqrt(2)*randn(m,n);
y = abs(A*x).^2 ;

%% Initialization

npower_iter = 50;                           % Number of power iterations 
z0 = randn(n,1); z0 = z0/norm(z0,'fro');    % Initial guess 
for tt = 1:npower_iter,                     % Power iterations
    z0 = A'*(y.* (A*z0)); z0 = z0/norm(z0,'fro');
end

normest = sqrt(sum(y)/numel(y));    % Estimate norm to scale eigenvector  
z = normest * z0;                   % Apply scaling 
Relerrs = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

%% Loop

T = 2500;                           % Max number of iterations
tau0 = 330;                         % Time constant for step size
mu = @(t) min(1-exp(-t/tau0), 0.2); % Schedule for step size

for t = 1:T,
    yz = A*z;
    grad  = 1/m* A'*( ( abs(yz).^2-y ) .* yz ); % Wirtinger gradient
    z = z - mu(t)/normest^2 * grad;             % Gradient update 

    Relerrs = [Relerrs, norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro')];  
end
 
%% Check results

 fprintf('Relative error after initialization: %f\n', Relerrs(1))
 fprintf('Relative error after %d iterations: %f\n', T, Relerrs(T+1))
 
 figure, semilogy(0:T,Relerrs) 
 xlabel('Iteration'), ylabel('Relative error (log10)'), ...
     title('Relative error vs. iteration count')
 
 

    
  
