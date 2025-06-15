% π值的蒙特卡洛估算（线性与对数间隔对比）
clc;
clear;
close all;

% 参数设置
num_points = 30;          % 数据点数量
num_experiments = 100;     % 每个N值重复实验次数
alpha = 0.05;             % 置信度95%
lambda_alpha = norminv(1-alpha/2); % 对应1.96

% 生成两种间隔的仿真次数
N_log = round(logspace(1, 6, num_points));    % 对数间隔：10^1到10^6
N_linear = round(linspace(100, 1e6, num_points)); % 线性间隔：100到1e6

% 初始化存储结构
results = struct();
spacings = {'log', 'linear'};

% 进行蒙特卡洛仿真（两种间隔）
for s = 1:length(spacings)
    spacing = spacings{s};
    if strcmp(spacing, 'log')
        N_list = N_log;
    else
        N_list = N_linear;
    end
    
    % 预分配存储
    pi_est = zeros(length(N_list), num_experiments);
    abs_err = zeros(length(N_list), 1);
    conf_int = zeros(length(N_list), 2);
    
    for i = 1:length(N_list)
        N = N_list(i);
        for j = 1:num_experiments
            % 随机撒点
            points = rand(N, 2);
            distances = sqrt(points(:,1).^2 + points(:,2).^2);
            inside = sum(distances <= 1);
            pi_est(i,j) = 4 * inside / N;
        end
        
        % 计算统计量
        mean_pi = mean(pi_est(i,:));
        std_pi = std(pi_est(i,:));
        abs_err(i) = lambda_alpha * std_pi / sqrt(num_experiments);
        conf_int(i,:) = [mean_pi - abs_err(i), mean_pi + abs_err(i)];
    end
    
    % 存储结果
    results.(spacing).N = N_list;
    results.(spacing).pi_est = pi_est;
    results.(spacing).abs_err = abs_err;
    results.(spacing).conf_int = conf_int;
end

% 绘制对比图
figure('Position', [100, 100, 1000, 800]);

% 1. π值估计对比
subplot(2,2,1); % 线性间隔π值
plot(results.linear.N, results.linear.pi_est, 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 3);
hold on;
plot(results.linear.N, mean(results.linear.pi_est,2), 'b-', 'LineWidth', 2);
plot([min(results.linear.N), max(results.linear.N)], [pi pi], 'r--', 'LineWidth', 2);
xlabel('仿真次数N');
ylabel('π估计值');
title('(a) 线性间隔π值估计');
legend('单次实验', '平均值', '真实值π', 'Location', 'southeast');
grid on;

subplot(2,2,2); % 对数间隔π值
semilogx(results.log.N, results.log.pi_est, 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 3);
hold on;
semilogx(results.log.N, mean(results.log.pi_est,2), 'b-', 'LineWidth', 2);
plot([min(results.log.N), max(results.log.N)], [pi pi], 'r--', 'LineWidth', 2);
xlabel('仿真次数N');
ylabel('π估计值');
title('(b) 对数间隔π值估计');
grid on;

% 2. 绝对误差对比
subplot(2,2,3); % 线性间隔误差
plot(results.linear.N, results.linear.abs_err, 'b-o', 'LineWidth', 1.5);
hold on;
plot(results.linear.N, 2.5./sqrt(results.linear.N), 'r--', 'LineWidth', 1.5);
xlabel('仿真次数N');
ylabel('绝对误差');
title('(c) 线性间隔绝对误差');
legend('实验误差', 'O(1/√N)参考', 'Location', 'northeast');
grid on;

subplot(2,2,4); % 对数间隔误差
loglog(results.log.N, results.log.abs_err, 'b-o', 'LineWidth', 1.5);
hold on;
loglog(results.log.N, 2.5./sqrt(results.log.N), 'r--', 'LineWidth', 1.5);
xlabel('仿真次数N');
ylabel('绝对误差');
title('(d) 对数间隔绝对误差');
grid on;

% 显示结果摘要
fprintf('对数间隔结果摘要:\n');
fprintf('%-10s %-10s %-10s %-15s\n', 'N', 'π估计', '绝对误差', '95%%置信区间');
disp_indices = [1, 10, 20, 30, 40, 50];
for i = disp_indices% π值的蒙特卡洛估算（线性与对数间隔对比）
clc;
clear;
close all;

% 参数设置
num_points = 30;          % 数据点数量
num_experiments = 50;     % 每个N值重复实验次数
alpha = 0.05;             % 置信度95%
lambda_alpha = norminv(1-alpha/2); % 对应1.96

% 生成两种间隔的仿真次数
N_log = round(logspace(1, 6, num_points));    % 对数间隔：10^1到10^6
N_linear = round(linspace(100, 1e6, num_points)); % 线性间隔：100到1e6

% 初始化存储结构
results = struct();
spacings = {'log', 'linear'};

% 进行蒙特卡洛仿真（两种间隔）
for s = 1:length(spacings)
    spacing = spacings{s};
    if strcmp(spacing, 'log')
        N_list = N_log;
    else
        N_list = N_linear;
    end
    
    % 预分配存储
    pi_est = zeros(length(N_list), num_experiments);
    abs_err = zeros(length(N_list), 1);
    conf_int = zeros(length(N_list), 2);
    
    for i = 1:length(N_list)
        N = N_list(i);
        for j = 1:num_experiments
            % 随机撒点
            points = rand(N, 2);
            distances = sqrt(points(:,1).^2 + points(:,2).^2);
            inside = sum(distances <= 1);
            pi_est(i,j) = 4 * inside / N;
        end
        
        % 计算统计量
        mean_pi = mean(pi_est(i,:));
        std_pi = std(pi_est(i,:));
        abs_err(i) = lambda_alpha * std_pi / sqrt(num_experiments);
        conf_int(i,:) = [mean_pi - abs_err(i), mean_pi + abs_err(i)];
    end
    
    % 存储结果
    results.(spacing).N = N_list;
    results.(spacing).pi_est = pi_est;
    results.(spacing).abs_err = abs_err;
    results.(spacing).conf_int = conf_int;
end

% 绘制对比图
figure('Position', [100, 100, 1000, 800]);

% 1. π值估计对比
subplot(2,2,1); % 线性间隔π值
plot(results.linear.N, results.linear.pi_est, 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 3);
hold on;
plot(results.linear.N, mean(results.linear.pi_est,2), 'b-', 'LineWidth', 2);
plot([min(results.linear.N), max(results.linear.N)], [pi pi], 'r--', 'LineWidth', 2);
xlabel('仿真次数N');
ylabel('π估计值');
title('(a) 线性间隔π值估计');
legend('单次实验', '平均值', '真实值π', 'Location', 'southeast');
grid on;

subplot(2,2,2); % 对数间隔π值
semilogx(results.log.N, results.log.pi_est, 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 3);
hold on;
semilogx(results.log.N, mean(results.log.pi_est,2), 'b-', 'LineWidth', 2);
plot([min(results.log.N), max(results.log.N)], [pi pi], 'r--', 'LineWidth', 2);
xlabel('仿真次数N');
ylabel('π估计值');
title('(b) 对数间隔π值估计');
grid on;

% 2. 绝对误差对比
subplot(2,2,3); % 线性间隔误差
plot(results.linear.N, results.linear.abs_err, 'b-o', 'LineWidth', 1.5);
hold on;
plot(results.linear.N, 2.5./sqrt(results.linear.N), 'r--', 'LineWidth', 1.5);
xlabel('仿真次数N');
ylabel('绝对误差');
title('(c) 线性间隔绝对误差');
legend('实验误差', 'O(1/√N)参考', 'Location', 'northeast');
grid on;

subplot(2,2,4); % 对数间隔误差
loglog(results.log.N, results.log.abs_err, 'b-o', 'LineWidth', 1.5);
hold on;
loglog(results.log.N, 2.5./sqrt(results.log.N), 'r--', 'LineWidth', 1.5);
xlabel('仿真次数N');
ylabel('绝对误差');
title('(d) 对数间隔绝对误差');
grid on;

% 显示结果摘要
disp_indices = [1, 7, 10, 15, 22, 30]; % 确保索引不超过数组长度

fprintf('对数间隔结果摘要:\n');
fprintf('%-10s %-10s %-10s %-15s\n', 'N', 'π估计', '绝对误差', '95%置信区间');
for i = disp_indices
    fprintf('%-10d %-10.4f %-10.4f [%-6.4f, %-6.4f]\n', ...
            results.log.N(i), mean(results.log.pi_est(i,:)), ...
            results.log.abs_err(i), results.log.conf_int(i,1), results.log.conf_int(i,2));
end

fprintf('\n线性间隔结果摘要:\n');
fprintf('%-10s %-10s %-10s %-15s\n', 'N', 'π估计', '绝对误差', '95%置信区间');
for i = disp_indices
    fprintf('%-10d %-10.4f %-10.4f [%-6.4f, %-6.4f]\n', ...
            results.linear.N(i), mean(results.linear.pi_est(i,:)), ...
            results.linear.abs_err(i), results.linear.conf_int(i,1), results.linear.conf_int(i,2));
end
    fprintf('%-10d %-10.4f %-10.4f [%-6.4f, %-6.4f]\n', ...
            results.log.N(i), mean(results.log.pi_est(i,:)), ...
            results.log.abs_err(i), results.log.conf_int(i,1), results.log.conf_int(i,2));
end

fprintf('\n线性间隔结果摘要:\n');
fprintf('%-10s %-10s %-10s %-15s\n', 'N', 'π估计', '绝对误差', '95%%置信区间');
for i = disp_indices
    fprintf('%-10d %-10.4f %-10.4f [%-6.4f, %-6.4f]\n', ...
            results.linear.N(i), mean(results.linear.pi_est(i,:)), ...
            results.linear.abs_err(i), results.linear.conf_int(i,1), results.linear.conf_int(i,2));
end