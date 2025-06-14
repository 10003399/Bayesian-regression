%%
% Uppgift 1
% === Parametrar ===
w = [0; 2.5; -0.5];   % Weight vector
sigma = [0.3, 0.5, 0.8, 1.2];  % Different noise levels

% === Create 2D grid ===
x1 = -1:0.05:1;
x2 = -1:0.05:1;
[X1, X2] = meshgrid(x1, x2);

% === Generate data ===
N = numel(X1);
X_design = [ones(N,1), X1(:).^2, X2(:).^3];  % Φ(X) as per Eq. 13
t = cell(1, length(sigma));

figure(1);
clf;

% Loop over different noise levels (sigma)
for i = 1:length(sigma)
    noise = normrnd(0, sigma(i), [N,1]);  % Use standard deviation, not variance
    t{i} = X_design * w + noise;

    % Create subplot and 3D axes
    ax = subplot(2,2,i);
    scatter3(X1(:), X2(:), t{i}, 10, t{i}, 'filled');
    
    xlabel('x_1');
    ylabel('x_2');
    zlabel('t');
    title(sprintf('Generated data (\\sigma = %.1f)', sigma(i)));
    
    view(3);        % Ensure 3D view
    grid on;
    axis vis3d;     % Maintain aspect ratio
    colormap parula;
    colorbar;
end


%%
% uppgift 2


x1_data = X1(:);
x2_data = X2(:);

sigma_extra = 0.3;

t_test_noisy = cell(1, length(sigma));
t_train = cell(1, length(sigma));
x1_test = cell(1, length(sigma));
x2_test = cell(1, length(sigma));
x1_train = cell(1, length(sigma));
x2_train = cell(1, length(sigma));


figure(2);
clf;

for i = 1:length(t)
    t_data = t{i};   % Vector of outcomes (1681x1)

    % Define training and testing regions
    is_test = (abs(x1_data) > 0.3) & (abs(x2_data) > 0.3);
    is_train = ~is_test;

    % Extract coordinates
    x1_test{i} = x1_data(is_test);
    x2_test{i} = x2_data(is_test);
    t_test = t_data(is_test);

    x1_train{i} = x1_data(is_train);
    x2_train{i} = x2_data(is_train);
    t_train{i} = t_data(is_train);

    % Add extra noise to test targets
    extra_noise = normrnd(0, sigma_extra, size(t_test));
    t_test_noisy{i} = t_test + extra_noise;

    % === Subplot for each sigma ===
    subplot(2,2,i);
    hold on;
    scatter3(x1_train{i}, x2_train{i}, t_train{i}, 10, 'r', 'filled');        % Red: training data
    scatter3(x1_test{i}, x2_test{i}, t_test_noisy{i}, 10, 'b', 'filled');     % Blue: test data

    xlabel('x_1');
    ylabel('x_2');
    zlabel('t');
    title(sprintf('Train/Test split (\\sigma = %.1f)', sigma(i)));

    view(3);
    grid on;
    axis vis3d;
end

%%
% Uppgift 3
% New figures for detailed 3D training and test prediction plots
figure; % For Training Predictions (3D)
fig_train = gcf;

figure; % For Test Predictions (3D)
fig_test = gcf;

figure; % For Beta_ML and MSE as originally intended
fig_summary = gcf;
MSE_ML_test = cell(1, length(sigma));

for i = 1:length(sigma)
    fprintf('--- Dataset %d ---\n', i);
    fprintf('Sigma (noise std): %.2f\n', sigma(i));
    % Training
    Phi_train = [ones(length(x1_train{i}), 1), x1_train{i}.^2, x2_train{i}.^3];
    w_ML = (Phi_train' * Phi_train) \ (Phi_train' * t_train{i});
    t_pred_train = Phi_train * w_ML;
    beta_inv = mean((t_train{i} - t_pred_train).^2);
    beta_ML = 1 / beta_inv;
    fprintf('Estimated weights (w_ML): [%.4f, %.4f, %.4f]\n', w_ML(1), w_ML(2), w_ML(3));
    fprintf('Estimated beta_ML (1/var): %.4f\n', beta_ML);

    % Test
    Phi_test = [ones(length(x1_test{i}), 1), x1_test{i}.^2, x2_test{i}.^3];
    t_pred_test = Phi_test * w_ML;
    t_true_test = w(1) + w(2)*x1_test{i}.^2 + w(3)*x2_test{i}.^3;
    MSE_ML_test{i} = mean((t_pred_test - t_true_test).^2);
    fprintf('Test MSE: %.4f\n\n', MSE_ML_test{i});

    % ---- 3D Figure for Training Predictions ----
    figure(fig_train);
    sgtitle('Training data prediction')
    subplot(2, 2, i);
    plot3(x1_train{i}, x2_train{i}, t_train{i}, 'bo', 'DisplayName', 'True'); hold on;
    plot3(x1_train{i}, x2_train{i}, t_pred_train, 'r.', 'DisplayName', 'Predicted');
    xlabel('x1'); ylabel('x2'); zlabel('t');
    title(sprintf('Train: \\sigma = %.2f', sigma(i)));
    legend('Location', 'best'); grid on; view(135, 25);

    % ---- 3D Figure for Test Predictions ----
    figure(fig_test);
    sgtitle('Test data prediction')
    subplot(2, 2, i);
    plot3(x1_test{i}, x2_test{i}, t_true_test, 'bo', 'DisplayName', 'True'); hold on;
    plot3(x1_test{i}, x2_test{i}, t_pred_test, 'r.', 'DisplayName', 'Predicted');
    xlabel('x1'); ylabel('x2'); zlabel('t');
    title(sprintf('Test: \\sigma = %.2f', sigma(i)));
    legend('Location', 'best'); grid on; view(135, 25);

    % ---- Summary Figure: Beta_ML and MSE (unchanged) ----
    figure(fig_summary);
    % Subplot 3: Beta_ML
    subplot(1,2,1)
    bar(sigma(i), beta_ML, 'BarWidth', 0.15); hold on;
    xlabel('Alpha value'); ylabel('\beta_{ML}');
    title('Estimated \beta_{ML} for Each Dataset');
    grid on;

    % Subplot 4: MSE
    subplot(1,2,2)
    bar(sigma(i), MSE_ML_test{i}, 'BarWidth', 0.15); hold on;
    xlabel('Alpha value'); ylabel('Mean Square Error');
    title('MSE on Test Sets');
    grid on;
end

% Adjust summary plots
figure(fig_summary);
subplot(1,2,1); xlim([0 max(sigma)*1.1]); hold off;
subplot(1,2,2); xlim([0 max(sigma)*1.1]); hold off;

%%
% Uppgift 4 - Utökad med loop över alpha och full Bayesian regression

alphas = [0.3, 0.7, 2.0];  % Three alpha values
[Xq1, Xq2] = meshgrid(linspace(-1, 1, 50), linspace(-1, 1, 50));
MSE_ALL = zeros(length(alphas), length(sigma));  % Store MSEs

for ai = 1:length(alphas)
    alpha = alphas(ai);
    
    % Create a large figure for all 12 subplots (4 sigmas × 3 plot types)
    figure('Name', sprintf('Results for alpha = %.1f', alpha));
    t = tiledlayout(4, 4, 'TileSpacing', 'Compact', 'Padding', 'Compact');
    title(t, sprintf('Results for \\alpha = %.1f', alpha), 'FontSize', 16, 'FontWeight', 'bold');

    for i = 1:length(sigma)
        beta = 1 / sigma(i)^2;

        % --- Data prep ---
        Phi_train = [ones(length(x1_train{i}), 1), x1_train{i}.^2, x2_train{i}.^3];
        Phi_test = [ones(length(x1_test{i}), 1), x1_test{i}.^2, x2_test{i}.^3];

        % --- Posterior ---
        A = alpha * eye(3) + beta * (Phi_train' * Phi_train);
        S_N = inv(A);
        m_N = beta * S_N * Phi_train' * t_train{i};

        % --- Prediction ---
        mean_pred = Phi_test * m_N;
        var_pred = zeros(length(x1_test{i}), 1);
        for j = 1:length(x1_test{i})
            phi_x = Phi_test(j, :)';
            var_pred(j) = (1/beta) + phi_x' * S_N * phi_x;
        end
        std_pred = sqrt(var_pred);

        % --- Ground truth and MSE ---
        t_true_test = w(1) + w(2)*x1_test{i}.^2 + w(3)*x2_test{i}.^3;
        MSE = mean((mean_pred - t_true_test).^2);
        MSE_ALL(ai, i) = MSE;
        fprintf('Alpha = %.2f | Sigma = %.1f | MSE = %.6f\n', alpha, sigma(i), MSE);

        % === Plot 1: 3D scatter of predictions ===
        nexttile((i - 1) * 4 + 1);
        scatter3(x1_test{i}, x2_test{i}, mean_pred, 20, std_pred, 'bo');
        hold on;
        scatter3(x1_test{i}, x2_test{i}, t_true_test, 10, 'r', 'filled');
        xlabel('x_1'); ylabel('x_2'); zlabel('t');
        title(sprintf('Prediction σ=%.2f', sigma(i)));
        legend('Predicted', 'True', 'Location', 'best');
        grid on; view(45, 25);

        % === Plot 2: Mean surface ===
        nexttile((i - 1) * 4 + 2);
        F_mean = scatteredInterpolant(x1_test{i}, x2_test{i}, mean_pred);
        MeanGrid = F_mean(Xq1, Xq2);
        surf(Xq1, Xq2, MeanGrid);
        title(sprintf('Mean Surface σ=%.2f', sigma(i)));
        xlabel('x_1'); ylabel('x_2'); zlabel('Predicted mean');
        shading interp; colorbar; grid on; view(45, 30);

        % === Plot 3: Uncertainty surface ===
        nexttile((i - 1) * 4 + 3);
        F_std = scatteredInterpolant(x1_test{i}, x2_test{i}, std_pred);
        StdGrid = F_std(Xq1, Xq2);
        surf(Xq1, Xq2, StdGrid);
        title(sprintf('Uncertainty σ=%.2f', sigma(i)));
        xlabel('x_1'); ylabel('x_2'); zlabel('Std deviation');
        shading interp; colorbar; grid on; view(45, 30);
    end

    % === Final row: Bar plot of MSEs ===
    nexttile(4, [4 1]);
    bar(sigma, MSE_ALL(ai, :), 0.5);
    title(sprintf('MSE (\\alpha = %.1f)', alpha));
    xlabel('\sigma');
    ylabel('MSE');
    grid on;
    ylim([0, max(MSE_ALL(ai, :)) * 1.1]);
end


%% Uppgift 5
figure('Name', 'All MSE Comparisons');

num_sigmas = length(sigma);
num_alphas = length(alphas);
MSE_combined = nan(num_sigmas, num_alphas + 1);

% Fill data
MSE_combined(:, 1:num_alphas) = MSE_ALL';  
for k = 1:num_sigmas
    MSE_combined(k, end) = MSE_ML_test{k};  
end

% Use horizontal layout: 1 row, multiple columns
for k = 1:num_sigmas
    subplot(1, num_sigmas, k);  % <-- horizontal layout
    
    bar(MSE_combined(k, :), 0.5);  % vertical bars
    
    title(sprintf('MSEs for \\sigma = %.2f', sigma(k)));
    xlabel('Model');
    ylabel('MSE');
    
    xticks(1:(num_alphas + 1));
    xticklabels([arrayfun(@(a) sprintf('\\  alpha = %.1f', a), alphas, 'UniformOutput', false), {'ML'}]);
    
    ylim([0 10^-2]);
    grid on;
end



%%
% Uppgift 6 - Utökad med loop över alpha och full Bayesian regression
alphas = [0.3, 0.7, 2.0];  % Testa flera alpha
sigma = [0.3, 0.5, 0.8, 1.2];
[Xq1, Xq2] = meshgrid(linspace(-1,1,50), linspace(-1,1,50));

% --- Preallocate result matrices: [sigma x alpha]
MSE_train_matrix = zeros(length(sigma), length(alphas));
MSE_test_matrix  = zeros(length(sigma), length(alphas));
var_train_matrix = zeros(length(sigma), length(alphas));
var_test_matrix  = zeros(length(sigma), length(alphas));

for i = 1:length(sigma)
    % Design matrices for current sigma
    Phi_train = [ones(length(x1_train{i}), 1), x1_train{i}.^2, x2_train{i}.^3];
    Phi_test  = [ones(length(x1_test{i}), 1),  x1_test{i}.^2,  x2_test{i}.^3];

    for ai = 1:length(alphas)
        alpha = alphas(ai);
        beta = 1 / sigma(i)^2;

        % --- Posteriorberäkning ---
        A = alpha * eye(3) + beta * (Phi_train' * Phi_train);
        S_N = inv(A);
        m_N = beta * S_N * Phi_train' * t_train{i};

        % --- Bayesian prediction (test data) ---
        mean_pred_test = Phi_test * m_N;
        var_pred_test = zeros(length(x1_test{i}), 1);
        for j = 1:length(x1_test{i})
            phi_x = Phi_test(j, :)';
            var_pred_test(j) = (1 / beta) + phi_x' * S_N * phi_x;
        end

        % --- Bayesian prediction (train data) ---
        mean_pred_train = Phi_train * m_N;
        var_pred_train = zeros(length(x1_train{i}), 1);
        for j = 1:length(x1_train{i})
            phi_x = Phi_train(j, :)';
            var_pred_train(j) = (1 / beta) + phi_x' * S_N * phi_x;
        end

        % --- True target values ---
        t_true_test  = w(1) + w(2) * x1_test{i}.^2  + w(3) * x2_test{i}.^3;
        t_true_train = w(1) + w(2) * x1_train{i}.^2 + w(3) * x2_train{i}.^3;

        % --- Metrics ---
        MSE_test  = mean((mean_pred_test  - t_true_test).^2);
        MSE_train = mean((mean_pred_train - t_true_train).^2);
        mean_var_test  = mean(var_pred_test);
        mean_var_train = mean(var_pred_train);

        % --- Save to matrices ---
        MSE_test_matrix(i, ai)  = MSE_test;
        MSE_train_matrix(i, ai) = MSE_train;
        var_test_matrix(i, ai)  = mean_var_test;
        var_train_matrix(i, ai) = mean_var_train;

        % --- Console output ---
        fprintf('Alpha = %.1f, Sigma = %.1f\n', alpha, sigma(i));
        fprintf('  MSE (train): %.6f, mean variance (train): %.6f\n', MSE_train, mean_var_train);
        fprintf('  MSE (test) : %.6f, mean variance (test) : %.6f\n\n', MSE_test, mean_var_test);
    end
end

% Convert sigma to sigma^2 (noise variance)
sigma2 = sigma.^2;

titles = arrayfun(@(a) sprintf('\\alpha = %.1f', a), alphas, 'UniformOutput', false);

% === Plot 1: MSE ===
figure;
for ai = 1:length(alphas)
    subplot(1, length(alphas), ai);
    
    mse_train = MSE_train_matrix(:, ai);
    mse_test  = MSE_test_matrix(:, ai);

    plot(sigma2, mse_train, '-o', 'LineWidth', 1.8, 'DisplayName', 'Train MSE');
    hold on;
    plot(sigma2, mse_test, '--o', 'LineWidth', 1.8, 'DisplayName', 'Test MSE');
    
    xlabel('\sigma^2');
    ylabel('MSE');
    title(['MSE for ', titles{ai}]);
    legend('Location', 'northwest');
    grid on;
end
sgtitle('Mean Squared Error vs \sigma^2 for Different \alpha');

% === Plot 2: Predictive Variance ===
figure;
for ai = 1:length(alphas)
    subplot(1, length(alphas), ai);
    
    var_train = var_train_matrix(:, ai);
    var_test  = var_test_matrix(:, ai);

    plot(sigma2, var_train, '-s', 'LineWidth', 1.8, 'DisplayName', 'Train Variance');
    hold on;
    plot(sigma2, var_test, '--s', 'LineWidth', 1.8, 'DisplayName', 'Test Variance');
    
    xlabel('\sigma^2');
    ylabel('Mean Predictive Variance');
    title(['Variance for ', titles{ai}]);
    legend('Location', 'northwest');
    grid on;
end
sgtitle('Predictive Variance vs \sigma^2 for Different \alpha');