% 加载数据（假设数据文件为data(3).txt）
x = load('data(3).txt');

% --- 数据预处理 ---
% 检查数据是否非负（瑞利分布要求x≥0）
if any(x < 0)
    error('数据包含负数，瑞利分布要求x≥0');
end

% --- 绘制柱状图与拟合曲线 ---
figure;
histogram(x, 'Normalization', 'pdf', 'BinMethod', 'sturges', 'EdgeColor', 'none');
hold on;
title('数据分布与瑞利分布拟合');
xlabel('x'); ylabel('概率密度');
grid on;

% --- 参数估计 ---
mu = mean(x);
sigma_hat = mu / sqrt(pi/2); % 矩估计公式
fprintf('估计参数σ = %.4f\n', sigma_hat);

% 绘制理论瑞利分布曲线
x_range = linspace(0, max(x), 1000);
pdf_rayleigh = (x_range / sigma_hat^2) .* exp(-x_range.^2 / (2*sigma_hat^2));
plot(x_range, pdf_rayleigh, 'r', 'LineWidth', 1.5);
legend('数据分布', '瑞利拟合', 'Location', 'northeast');

% --- 卡方检验优化部分 ---
% 自动分箱（Sturges规则）
k_initial = ceil(log2(numel(x))) + 1;
[observed, edges] = histcounts(x, k_initial);

% 计算期望频数（向量化替代循环）
cdf_values = 1 - exp(-edges.^2 / (2*sigma_hat^2)); % 直接使用瑞利CDF公式
expected = numel(x) * diff(cdf_values);

% 合并低期望区间（封装合并逻辑）
[observed_adj, expected_adj] = mergeLowFrequencyBins(observed, expected, 5);

% 处理合并后区间不足的情况
if numel(expected_adj) < 2
    k_initial = max(2, floor(k_initial/2));
    [observed, edges] = histcounts(x, k_initial);
    expected = numel(x) * diff(1 - exp(-edges.^2/(2*sigma_hat^2)));
    [observed_adj, expected_adj] = mergeLowFrequencyBins(observed, expected, 5);
end

% 剔除无效区间（期望为0）
valid_bins = expected_adj > 0;
observed_adj = observed_adj(valid_bins);
expected_adj = expected_adj(valid_bins);

% 计算卡方统计量与自由度
chi2_stat = sum((observed_adj - expected_adj).^2 ./ expected_adj);
df = numel(observed_adj) - 1 - 1; % 自由度 = 区间数 - 参数数 -1
alpha = 0.05;
critical_value = chi2inv(1 - alpha, df);

% 输出结果
fprintf('卡方统计量: %.4f\n', chi2_stat);
fprintf('自由度: %d\n', df);
fprintf('临界值(α=0.05): %.4f\n', critical_value);
if chi2_stat > critical_value
    disp('结论: 拒绝原假设，数据不服从瑞利分布');
else
    disp('结论: 无法拒绝原假设，数据可能服从瑞利分布');
end

% --- 子函数：合并低期望频数区间 ---
function [obs_adj, exp_adj] = mergeLowFrequencyBins(obs, exp, min_exp)
    obs_adj = [];
    exp_adj = [];
    current_obs = 0;
    current_exp = 0;
    
    for i = 1:numel(exp)
        current_obs = current_obs + obs(i);
        current_exp = current_exp + exp(i);
        % 若当前累积期望≥阈值或到最后一个区间，则合并
        if current_exp >= min_exp || i == numel(exp)
            obs_adj = [obs_adj, current_obs];
            exp_adj = [exp_adj, current_exp];
            current_obs = 0;
            current_exp = 0;
        end
    end
end