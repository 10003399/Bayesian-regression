

w0 = -1.2;
w1 = 0.9;

trnX = -1:0.01:1;
tstX = -1.5:0.1:1.5;

mu = 0;
sigma2 = 0.2;


% Generera träningsdata
count = 0;
for x = trnX
    count = count + 1;
    e = normrnd(mu, sigma2);   % dra ny brus för varje datapunkt
    trnData(count) = w0 + w1*x + e;
end

% Generera testdata
counttrn = 0;
for x = tstX
    counttrn = counttrn + 1;
    e = normrnd(mu, sigma2);   % dra ny brus för varje datapunkt
    tstData(counttrn) = w0 + w1*x + e;
end

totData = [tstData, trnData];

figure;
hold on;
scatter(trnX, trnData, 'b.');
scatter(tstX, tstData, 'r.');
plot(trnX, w0 + w1*trnX, 'k-', 'LineWidth', 3);  % Sanna linjen
xlabel('x');
ylabel('t');
legend('Träningsdata', 'Testdata', 'Sann linje');
title('Genererad tränings- och testdata');
grid on;

%%
%prior

% === Prior-parametrar ===
alpha = 2;
Sigma_prior = (1/alpha) * eye(2);  % Kovariansmatris: 0.5*I
mu_prior = [0; 0];

% === Skapa meshgrid (samma som Python-exemplet) ===
w0_list = linspace(-2, 2, 200);
w1_list = linspace(-2, 2, 200);
[W0, W1] = meshgrid(w0_list, w1_list);

% === Vektorisera punkterna ===
points = [W0(:), W1(:)];

% === Beräkna prior density för varje punkt ===
prior_pdf = mvnpdf(points, mu_prior', Sigma_prior);

% === Omforma tillbaka till samma storlek som W0/W1 ===
prior_pdf = reshape(prior_pdf, size(W0));

% === Plot ===
figure;
contour(W0, W1, prior_pdf, 30);  % 30 nivåer
xlabel('w_0');
ylabel('w_1');
title('Multivariate Gaussian Prior (Contour Plot)');
colorbar;

hold on;
plot(-1.2, 0.9, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  % Exempel: markera w = [-1.2; 0.9]

%% Likelihood
trnX = -1:0.01:1;

% Välj ett subset av träningsdata (t.ex. 3 punkter för första modellen)
subset_idx = randperm(length(trnX), 201);
x_subset = trnX(subset_idx);
t_subset = trnData(subset_idx);

% Skapa meshgrid för w0 och w1 (samma som tidigare)
[W0, W1] = meshgrid(w0_list, w1_list);
likelihood_pdf = zeros(size(W0));

% Parametrar för noise
beta = 1 / sigma2;  % eftersom sigma2 är variansen

% Loop över hela w0-w1 grid
for i = 1:numel(W0)
    w = [W0(i); W1(i)];
    likelihood = 1;
    for j = 1:length(x_subset)
        x_vec = [1; x_subset(j)];  % Bias + x
        mean_pred = w' * x_vec;
        % Sannolikheten för t_j givet w och x_j (normalfördelning)
        likelihood_j = normpdf(t_subset(j), mean_pred, sqrt(1/beta));
        likelihood = likelihood * likelihood_j;
    end
    likelihood_pdf(i) = likelihood;
end

% Reshape för contour plot
likelihood_pdf = reshape(likelihood_pdf, size(W0));

% Plot likelihood
figure;
contour(W0, W1, likelihood_pdf, 30);
xlabel('w_0');
ylabel('w_1');
title('Likelihood (Contour Plot)');
colorbar;
hold on;
plot(-1.2, 0.9, 'rx', 'MarkerSize', 10, 'LineWidth', 2);  % True w


%% Uppgift 3
% Definiera parametrar
w0 = -1.2;
w1 = 0.9;
mu = 0;
sigma2 = 0.2;
alpha = 2;
beta = 1 / sigma2;

trnX = -1:0.01:1;
trnData = zeros(size(trnX));
for i = 1:length(trnX)
    e = normrnd(mu, sigma2);
    trnData(i) = w0 + w1 * trnX(i) + e;
end

subset_idx = randperm(length(trnX), 201);
x_subset = trnX(subset_idx);
t_subset = trnData(subset_idx);

% Skapa matris X med koordinater (1, x)
X = [ones(length(x_subset), 1), x_subset'];

% Posterior medelvärde och kovarians
S_n = inv(alpha * eye(2) + beta * (X') * X);
m_n = beta * S_n * (X') * t_subset';

% Posterior medelvärde och kovarians är redan beräknade: m_n, S_n

% Skapa grid av w0 och w1 för contour-plot
w0_vals = linspace(-1.35, -1.05, 200);
w1_vals = linspace(0.7, 1.1, 200);
[W0_grid, W1_grid] = meshgrid(w0_vals, w1_vals);

% Beräkna posterior-PDF för varje punkt i grid: använd mvnpdf
posterior_pdf = zeros(size(W0_grid));
for i = 1:numel(W0_grid)
    w = [W0_grid(i), W1_grid(i)];
    posterior_pdf(i) = mvnpdf(w, m_n', S_n);
end

% Rita contour-plot
figure;
contour(W0_grid, W1_grid, posterior_pdf, 50, 'LineWidth', 1.2);
hold on;

% Plot posterior-medelvärde som en röd punkt
plot(m_n(1), m_n(2), 'ro', 'MarkerFaceColor','r');

% Etiketter och titel
xlabel('w_0');
ylabel('w_1');
title('Posterior PDF för vikter (w_0, w_1)');
grid on;
colorbar;
axis equal;

%% Uppgift 4 - Korrekt generering och kombination av data

% Definiera parametrar
w0 = -1.2;
w1 = 0.9;
mu = 0;
sigma2 = 0.2;
alpha = 2;
beta = 1 / sigma2;

% Definiera tränings- och testintervall
trnX = -1:0.01:1;
tstX = -1.5:0.1:1.5;

% Generera träningsdata
trnData = zeros(size(trnX));
for i = 1:length(trnX)
    e = normrnd(mu, sigma2);
    trnData(i) = w0 + w1 * trnX(i) + e;
end

% Generera testdata
tstData = zeros(size(tstX));
for i = 1:length(tstX)
    e = normrnd(mu, sigma2);
    tstData(i) = w0 + w1 * tstX(i) + e;
end

% Kombinera data (för att analysera hela datasetet)
totX = [trnX, tstX];
totData = [trnData, tstData];

% Slumpmässigt val av subset för posterioranalys (ta exempelvis 201 punkter från totala datan)
subset_idx = randperm(length(totX), 231);
x_subset = totX(subset_idx);
t_subset = totData(subset_idx);

% Skapa matris X med koordinater (1, x)
X = [ones(length(x_subset), 1), x_subset'];

% Posterior medelvärde och kovarians
S_n = inv(alpha * eye(2) + beta * (X') * X);
m_n = beta * S_n * (X') * t_subset';

% Välj 5 slumpmässiga viktvektorer från posteriorn
post_vectors = mvnrnd(m_n', S_n, 5);

% Plot
figure;
hold on;
scatter(trnX, trnData, 'b.', 'DisplayName','Training Data');
scatter(tstX, tstData, 'r.', 'DisplayName','Test Data');

% Generera och plotta posterior-samples
x_vals = linspace(-2, 2, 100);
for i = 1:5
    y_vals = post_vectors(i,1) + post_vectors(i,2) * x_vals;
    plot(x_vals, y_vals, 'LineWidth', 1.5, 'DisplayName',sprintf('Sample %d', i));
end

% Plotta den sanna linjen
y_true = w0 + w1 * x_vals;
plot(x_vals, y_true, 'k--', 'LineWidth', 2, 'DisplayName','True Line');

xlabel('x');
ylabel('t');
title('Posterior Samples and Combined Data');
grid on;
legend('show');

%% Uppgift 5
w0 = -1.2;
w1 = 0.9;
mu = 0;
sigma2 = 0.2;
alpha = 2;
beta = 1 / sigma2;

tstX = -1.5:0.1:1.5;
tstData = zeros(size(tstX));
for i = 1:length(tstX)
    e = normrnd(mu, sigma2);
    tstData(i) = w0 + w1 * tstX(i) + e;
end

% Kombinera träningsdata med testdata
trnX = -1:0.01:1;
trnData = zeros(size(trnX));
for i = 1:length(trnX)
    e = normrnd(mu, sigma2);
    trnData(i) = w0 + w1 * trnX(i) + e;
end
totX = [trnX, tstX];
totData = [trnData, tstData];

% Välj subset för posterioranalys
subset_idx = randperm(length(totX), 31);
x_subset = totX(subset_idx);
t_subset = totData(subset_idx);

X = [ones(length(x_subset), 1), x_subset'];

S_n = inv(alpha * eye(2) + beta * (X') * X);
m_n = beta * S_n * (X') * t_subset';

% Förutsägelse för varje testpunkt
pred_mean = zeros(size(tstX));
pred_std = zeros(size(tstX));
for i = 1:length(tstX)
    x_vec = [1; tstX(i)];
    pred_mean(i) = m_n' * x_vec;
    pred_std(i) = sqrt(1/beta + x_vec' * S_n * x_vec);
end

w_ML = inv(X' * X) * (X' * t_subset');
max_lh_pred = zeros(size(tstX));
for i = 1:length(tstX)
    x_vec = [1; tstX(i)];
    max_lh_pred(i) = w_ML' * x_vec;
end


% Plot
figure;
hold on;
errorbar(tstX, pred_mean, pred_std, 'bo', 'LineWidth', 1.2);
scatter(tstX, tstData, 'rx');
plot(tstX, max_lh_pred, 'g-', 'LineWidth', 1.5);
xlabel('Testpunkt X');
ylabel('Förutsagd output');
title('Predikterat medelvärde och standardavvikelse för testpunkter');
grid on;
legend('Prediktion med std','Verklig data','Maximum likelihood estimator','Location','Best');


