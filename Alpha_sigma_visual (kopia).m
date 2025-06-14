% === Inställningar ===
w0 = -1.2;
w1 = 0.9;

sigma2_list = [0.1, 0.4, 0.8];  % Olika brusnivåer
alpha_list = [0.5, 2, 5];       % Olika prior-precisioner

trnX = -1:0.1:1;
N = length(trnX);
x_vals = linspace(-1.5, 1.5, 100);

% === Figur för brusnivåer ===
figure;
subplot(1,2,1);
hold on;

for idx = 1:length(sigma2_list)
    sigma2 = sigma2_list(idx);
    beta = 1 / sigma2;
    alpha = 2;  % Fixera alpha här

    % Skapa träningsdata
    trnData = zeros(size(trnX));
    for i = 1:N
        e = normrnd(0, sqrt(sigma2));
        trnData(i) = w0 + w1 * trnX(i) + e;
    end

    % Posterior
    X = [ones(N, 1), trnX'];
    S_n = inv(alpha * eye(2) + beta * X' * X);
    m_n = beta * S_n * X' * trnData';

    % Prediktion
    mu = zeros(size(x_vals));
    std_dev = zeros(size(x_vals));
    for i = 1:length(x_vals)
        x_vec = [1; x_vals(i)];
        mu(i) = m_n' * x_vec;
        std_dev(i) = sqrt(1/beta + x_vec' * S_n * x_vec);
    end

    % Rita prediktion + osäkerhet
    plot(x_vals, mu, 'LineWidth', 1.5, 'DisplayName', sprintf('\\sigma^2 = %.1f', sigma2));
    fill([x_vals, fliplr(x_vals)], ...
         [mu - std_dev, fliplr(mu + std_dev)], ...
         'k', 'FaceAlpha', 0.1 * idx, 'EdgeColor', 'none');  % Lite genomskinlig fyllning
end

% Rita träningsdata (från sista körningen)
scatter(trnX, trnData, 20, 'k', 'filled', 'DisplayName', 'Training data');

% Sanna linjen
y_true = w0 + w1 * x_vals;
plot(x_vals, y_true, 'k--', 'LineWidth', 2, 'DisplayName', 'True line');

xlabel('x');
ylabel('t');
title('Effekt av olika \\sigma^2 (brusnivå)');
legend('show');
grid on;


% === Figur för alpha (priorprecision) ===
subplot(1,2,2);
hold on;

sigma2 = 0.2;  % Fixera sigma2 här
beta = 1 / sigma2;

for idx = 1:length(alpha_list)
    alpha = alpha_list(idx);

    % Skapa träningsdata (samma brusnivå)
    trnData = zeros(size(trnX));
    for i = 1:N
        e = normrnd(0, sqrt(sigma2));
        trnData(i) = w0 + w1 * trnX(i) + e;
    end

    % Posterior
    X = [ones(N, 1), trnX'];
    S_n = inv(alpha * eye(2) + beta * X' * X);
    m_n = beta * S_n * X' * trnData';

    % Prediktion
    mu = zeros(size(x_vals));
    std_dev = zeros(size(x_vals));
    for i = 1:length(x_vals)
        x_vec = [1; x_vals(i)];
        mu(i) = m_n' * x_vec;
        std_dev(i) = sqrt(1/beta + x_vec' * S_n * x_vec);
    end

    % Rita prediktion + osäkerhet
    plot(x_vals, mu, 'LineWidth', 1.5, 'DisplayName', sprintf('\\alpha = %.1f', alpha));
    fill([x_vals, fliplr(x_vals)], ...
         [mu - std_dev, fliplr(mu + std_dev)], ...
         'k', 'FaceAlpha', 0.1 * idx, 'EdgeColor', 'none');
end

% Rita träningsdata (från sista körningen)
scatter(trnX, trnData, 20, 'k', 'filled', 'DisplayName', 'Training data');

% Sanna linjen
plot(x_vals, y_true, 'k--', 'LineWidth', 2, 'DisplayName', 'True line');

xlabel('x');
ylabel('t');
title('Effekt av olika \\alpha (prior precision)');
legend('show');
grid on;
