%% =========================================================================================== %%
%%                               Time-Varying Graphical Lasso Project
%% =========================================================================================== %%

%% ------------------------------------------- Data Import ----------------------------------- %%

% To import the data, simply drag or place the .csv file into the Current Folder"
data_pure = readtable('stock_data49.csv');
prices = data_pure;

%% --------------------------------- Verification & Preprocessing ---------------------------- %%

% Verify that the variable 'prices' is loaded
if ~exist('prices', 'var')
    error('Prices data is not loaded in the workspace.');
end

% Extract the dates and price data
if istable(prices)
    if ismember('Date', prices.Properties.VariableNames)
        dates = prices.Date;            % Extract dates
        priceData = prices{:, 2:end};   % Extract numeric columns
    else
        error('The table does not contain a "Date" column.');
    end
elseif isnumeric(prices) && ismatrix(prices)
    priceData = prices;   % If 'prices' is already a numeric matrix
else
    error('Prices data format is invalid.');
end

% Check for NaN values
if any(isnan(priceData), 'all')
    error('Price data contains NaN values. Please clean the dataset.');
end

% Compute log-returns
logReturns = diff(log(priceData));

% Center the log-returns
meanLogReturns = mean(logReturns, 1);
centeredLogReturns = logReturns - meanLogReturns;

[numSamples, numFeatures] = size(centeredLogReturns);

% Basic check: numSamples must be > numFeatures
if numSamples < numFeatures
    error('Insufficient samples for TVGL. Ensure numSamples > numFeatures.');
end


%% ----------------------------------- TVGL Parameters --------------------------------------- %%
lambda = 2.5;            % Sparsity parameter
beta = 10;               % Temporal regularization parameter
rho = 0.5;               % ADMM penalty parameter
maxIter = 100;           % Maximum number of iterations
tol = 0.01;              % Convergence tolerance
penaltyType = 'L1';      % Penalty type: 'L1', 'L2', 'Laplacian', 'Linf'


%% ------------------------------- Call to the TVGL function --------------------------------- %%
% thetaSet: Cell array of precision matrices (one per time slice)
thetaSet = TVGL(centeredLogReturns, lambda, beta, rho, maxIter, tol, penaltyType);


%% -------------------------------- Graphical Visualization ---------------------------------- %%
% Compute correlation among the Assets
corrMatrix = corrcoef(priceData);

% Plot the heatmap
figure;
heatmap(corrMatrix);
title('Heatmap of the Assets');
xlabel('Assets');
ylabel('Assets');

% Force the color limits from -1 (perfect negative correlation) to +1 (perfect positive)
h.ColorLimits = [-1 1];

% Define a custom color scale:
%  - Dark red for -1
%  - White for 0
%  - Dark green for +1
cVals = [-1 0 1];
cMapRGB = [
    0.5  0    0;  % dark red for -1
    1.0  1.0  1.0;  % white for  0
    0    0.5  0   % dark green for +1
];

% Number of interpolation points (controls smoothness)
numInterpol = 256;  
xq = linspace(-1, 1, numInterpol);

% Interpolate each color channel across -1 to +1
r = interp1(cVals, cMapRGB(:,1), xq, 'linear');
g = interp1(cVals, cMapRGB(:,2), xq, 'linear');
b = interp1(cVals, cMapRGB(:,3), xq, 'linear');

% Combine into an Nx3 colormap
customCMap = [r(:), g(:), b(:)];

% Apply the custom colormap to the current figure
colormap(customCMap);

% 1) First time slice
t1 = 1;
theta1 = thetaSet{t1};
edges1 = abs(theta1) > 1e-6;
[nodeIdx1_1, nodeIdx2_1] = find(triu(edges1, 1));
G1 = graph(nodeIdx1_1, nodeIdx2_1);

figure;
plot(G1);
title(sprintf('Graphical Representation (Time Slice %d)', t1));
xlabel('Assets');
ylabel('Assets');

% 2) Middle time slice
t_mid = floor(length(thetaSet) / 2);
theta_mid = thetaSet{t_mid};
edges_mid = abs(theta_mid) > 1e-6;
[nodeIdx1_mid, nodeIdx2_mid] = find(triu(edges_mid, 1));
G_mid = graph(nodeIdx1_mid, nodeIdx2_mid);

figure;
plot(G_mid);
title(sprintf('Graphical Representation (Time Slice %d)', t_mid));
xlabel('Assets');
ylabel('Assets');

% 3) Last time slice
t_last = length(thetaSet);
theta_last = thetaSet{t_last};
edges_last = abs(theta_last) > 1e-6;
[nodeIdx1_last, nodeIdx2_last] = find(triu(edges_last, 1));
G_last = graph(nodeIdx1_last, nodeIdx2_last);

figure;
plot(G_last);
title(sprintf('Graphical Representation (Time Slice %d)', t_last));
xlabel('Assets');
ylabel('Assets');

% Save the precision matrices
save('thetaSet.mat', 'thetaSet');
fprintf('Graphs plotted for time slices %d, %d, and %d.\n', t1, t_mid, t_last);
fprintf('Information matrices (thetaSet) saved to "thetaSet.mat".\n');

% Compute temporal deviation of consecutive precision matrices
numSlices = length(thetaSet);
temporalDeviation = zeros(1, numSlices - 1);

for t = 2:numSlices
    Theta_t = thetaSet{t};
    Theta_prev = thetaSet{t-1};
    temporalDeviation(t-1) = norm(Theta_t - Theta_prev, 'fro');  % Frobenius norm
end

% Plot the temporal deviation
figure;
plot(2:numSlices, temporalDeviation, '-o', 'LineWidth', 1.5);
xlabel('Time Slice');
ylabel('Deviation (Frobenius Norm)');
title('Temporal Deviation of Precision Matrices');
grid on;

%% ---------------------------- Portfolio Optimization --------------------------------------- %%
% Compute and visualize the evolution of the minimum variance portfolio
% weights over each time slice

if ~exist('thetaSet','var')
    error('Precision matrices (thetaSet) are not loaded in the workspace.');
end

weightsEvolving = zeros(numFeatures, length(thetaSet));

for ti = 1:length(thetaSet)
    theta_t = thetaSet{ti};        % Precision matrix at time slice ti
    oneVec = ones(numFeatures, 1);
    thetaOne = theta_t * oneVec;
    weights = thetaOne / (oneVec' * thetaOne);  % w = (Theta * 1) / (1' * Theta * 1)
    weightsEvolving(:, ti) = weights;
end

% Plot the evolution of the weights
figure;
hold on;
for i = 1:numFeatures
    plot(1:length(thetaSet), weightsEvolving(i, :), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Asset %d', i));
end
title('Portfolio Weight Evolution (Minimum Variance)', 'FontSize', 14);
xlabel('Time Slice', 'FontSize', 12);
ylabel('Portfolio Weight', 'FontSize', 12);
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
hold off;

% Save the weights
save('weightsEvolving.mat', 'weightsEvolving');
fprintf('Minimum variance portfolio weights saved to "weightsEvolving.mat".\n');


%% ======================================= TVGL Function ===================================== %%
function thetaSet = TVGL(data, lambda, beta, rho, maxIter, tol, penaltyType)
    % TVGL: Time-Varying Graphical Lasso Algorithm with explicit ADMM updates
    % Inputs:
    %   - data: Multivariate data matrix (num_samples x num_features)
    %   - lambda: Sparsity penalty parameter
    %   - beta: Temporal consistency parameter
    %   - rho: ADMM penalty parameter
    %   - maxIter: Maximum ADMM iterations
    %   - tol: Convergence tolerance
    %   - penaltyType: Type of penalty ('L1', 'L2', 'Laplacian', 'Linf')
    % Outputs:
    %   - thetaSet: Cell array of precision matrices (\Theta) per time slice

    % Dimensions
    [numSamples, numFeatures] = size(data);
    numTimeSlices = floor(numSamples / numFeatures);

    % Precompute empirical covariance matrices
    empCovSet = cell(1, numTimeSlices);
    for t = 1:numTimeSlices
        % Partition the data as in your original code
        empCovSet{t} = cov(data((t-1)*numFeatures+1 : t*numFeatures, :));
    end

    % Initialize variables
    thetaSet = cell(1, numTimeSlices);
    Z = cell(3, numTimeSlices);
    U = cell(3, numTimeSlices);
    for t = 1:numTimeSlices
        thetaSet{t} = eye(numFeatures);
        Z{1, t} = zeros(numFeatures);
        Z{2, t} = zeros(numFeatures);
        Z{3, t} = zeros(numFeatures);
        U{1, t} = zeros(numFeatures);
        U{2, t} = zeros(numFeatures);
        U{3, t} = zeros(numFeatures);
    end

    % =================== ADMM iterations ================== %
    for iter = 1:maxIter
        
        % ----------------- (1) Theta-Update ---------------- %
        for t = 1:numTimeSlices
            A = (Z{1, t} + Z{2, t} + Z{3, t} - U{1, t} - U{2, t} - U{3, t}) / 3;
            A = 0.5 * (A + A');  % Symmetrize
            eta = numFeatures / (3 * rho);
            S = empCovSet{t};

            % Eigendecomposition and analytical solution
            [Q, D] = eig(eta^(-1) * A - S);
            D = diag(D);
            D_new = (D + sqrt(D.^2 + 4 * eta^(-1))) / (2 * eta^(-1));
            thetaSet{t} = Q * diag(D_new) * Q'; % Reconstruct Theta
        end

        % ----------------- (2-1) Z0-Update ----------------- %
        for t = 1:numTimeSlices
            A = thetaSet{t} + U{1, t};
            lambda_rho = lambda / rho;

            % Element-wise soft-thresholding
            for i = 1:numFeatures
                for j = 1:numFeatures
                    if i ~= j
                        if abs(A(i, j)) <= lambda_rho
                            Z{1, t}(i, j) = 0;
                        else
                            Z{1, t}(i, j) = sign(A(i, j)) * (abs(A(i, j)) - lambda_rho);
                        end
                    else
                        Z{1, t}(i, j) = A(i, j);
                    end
                end
            end
        end

        % -------------- (2-2) (Z_1, Z_2)-Update ------------- %
        for t = 2:numTimeSlices
            % Compute A for E
            A = (thetaSet{t-1} - thetaSet{t} + U{2, t-1} - U{3, t});
            eta_2 = 2 * beta / rho;

            % Choose penalty
            switch penaltyType
                case 'L1'
                    E = sign(A) .* max(abs(A) - eta_2, 0);

                case 'L2'
                    normA = sqrt(sum(A.^2, 2));
                    scaling = max(0, 1 - eta_2 ./ normA);
                    E = bsxfun(@times, A, scaling);

                case 'Laplacian'
                    E = (1 / (1 + 2 * eta_2)) * A;

                case 'Linf'
                    normA = sum(abs(A), 2);
                    E = zeros(size(A));
                    for j = 1:size(A, 1)
                        if normA(j) <= eta_2
                            E(j, :) = 0;
                        else
                            % Bisection method to find sigma
                            sigma_lower = 0;
                            sigma_upper = max(abs(A(j, :))) / eta_2;
                            tol_linf = 1e-4;
                            while sigma_upper - sigma_lower > tol_linf
                                sigma = (sigma_upper + sigma_lower) / 2;
                                if sum(max(abs(A(j, :)) / eta_2 - sigma, 0)) > 1
                                    sigma_lower = sigma;
                                else
                                    sigma_upper = sigma;
                                end
                            end
                            sigma = (sigma_upper + sigma_lower) / 2;
                            E(j, :) = A(j, :) - eta_2 * max(abs(A(j, :)) / eta_2 - sigma, 0) .* sign(A(j, :));
                        end
                    end

                otherwise
                    error('Unsupported penalty type.');
            end

            % Update Z{2} and Z{3}
            Z{2, t-1} = 0.5 * (thetaSet{t-1} + thetaSet{t} + U{2, t-1} + U{3, t}) + 0.5 * (-E);
            Z{3, t}   = 0.5 * (thetaSet{t-1} + thetaSet{t} + U{2, t-1} + U{3, t}) + 0.5 * E;
        end

        % -------------- Update the U variables ------------- %
        for t = 1:numTimeSlices
            U{1, t} = U{1, t} + (thetaSet{t} - Z{1, t});
            if t < numTimeSlices
                U{2, t}   = U{2, t} + (thetaSet{t}   - Z{2, t});
                U{3, t+1} = U{3, t+1} + (thetaSet{t+1} - Z{3, t+1});
            end
        end

        % -------------- Convergence check ------------------ %
        [primalRes, dualRes] = ComputeResiduals(thetaSet, Z, U);
        if primalRes < tol && dualRes < tol
            fprintf('Outer ADMM converged at iteration %d\n', iter);
            break;
        end
    end

    % ================= Sub-function: Residuals ================ %
    function [primalRes, dualRes] = ComputeResiduals(thetaSet_, Z_, U_)
        primalRes = 0;
        dualRes = 0;

        for z = 1:length(thetaSet_)
            % Primal residual for Z{1}
            primalRes = primalRes + norm(thetaSet_{z} - Z_{1, z}, 'fro');

            % Dual residual for Z{2} and Z{3} (for t>1)
            if z > 1
                dualRes = dualRes + norm(Z_{2, z-1} - Z_{3, z}, 'fro');
            end
        end

        % Normalization
        primalRes = primalRes / length(thetaSet_);
        if length(thetaSet_) > 1
            dualRes = dualRes / (length(thetaSet_) - 1);
        end
    end
end
