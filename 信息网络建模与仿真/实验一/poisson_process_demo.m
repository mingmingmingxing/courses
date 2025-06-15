function poisson_process_demo()
    % 参数设置
    lambda = 2;         % 泊松过程率参数(事件/秒)
    T = 1000;           % 总观察时间(秒)
    
    % 生成泊松过程事件时间
    event_times = generate_poisson_process(lambda, T);
    
    % 统计每秒事件数
    counts_per_second = histcounts(event_times, 0:T);
    
    % 绘制事件时间序列
    figure;
    subplot(3,1,1);
    stem(event_times, ones(size(event_times)), 'Marker', 'none');
    xlabel('时间 (秒)');
    ylabel('事件发生');
    title(sprintf('λ=%.1f的泊松过程事件序列', lambda));
    xlim([0 50]); % 只显示前50秒
    
    % 绘制事件计数直方图
    subplot(3,1,2);
    max_count = max(counts_per_second);
    histogram(counts_per_second, 'BinEdges', -0.5:1:(max_count+0.5), 'Normalization', 'probability');
    hold on;
    
    % 绘制理论泊松分布
    k = 0:max_count;
    poisson_probs = poisspdf(k, lambda);
    stem(k, poisson_probs, 'r', 'LineWidth', 2);
    
    xlabel('每秒事件数');
    ylabel('概率');
    title('单位时间事件计数分布');
    legend('实际统计', '理论泊松分布');
    
    % 进行卡方检验 - 修正后的实现
    observed_counts = histcounts(counts_per_second, -0.5:1:(max_count+0.5));
    expected_counts = poisspdf(0:max_count, lambda) * T;
    
    % 确保没有零期望值
    valid_bins = expected_counts >= 5;
    if sum(valid_bins) < 2
        error('样本量不足，无法进行有效的卡方检验');
    end
    
    observed = observed_counts(valid_bins);
    expected = expected_counts(valid_bins);
    
    % 手动计算卡方统计量和p值
    chi2stat = sum((observed - expected).^2 ./ expected);
    df = length(observed) - 1 - 1; % 自由度 = 组数 - 1 - 估计参数数
    p = 1 - chi2cdf(chi2stat, df);
    
    % 显示检验结果
    subplot(3,1,3);
    axis off;
    text(0.1, 0.8, sprintf('泊松过程验证结果 (λ=%.1f, T=%.0f秒)', lambda, T), 'FontSize', 12);
    text(0.1, 0.6, sprintf('总事件数: %d', length(event_times)), 'FontSize', 12);
    text(0.1, 0.4, sprintf('平均事件率: %.3f 事件/秒', length(event_times)/T), 'FontSize', 12);
    text(0.1, 0.2, sprintf('卡方检验p值: %.4f', p), 'FontSize', 12);
    
    if p > 0.05
        text(0.1, 0.0, '结论: 数据符合泊松分布 (p > 0.05)', 'FontSize', 12, 'Color', 'green');
    else
        text(0.1, 0.0, '结论: 数据不符合泊松分布 (p ≤ 0.05)', 'FontSize', 12, 'Color', 'red');
    end
end

function event_times = generate_poisson_process(lambda, T)
    % 生成泊松过程事件时间
    % 使用指数分布间隔时间方法
    
    event_times = [];
    current_time = 0;
    
    while current_time < T
        % 生成下一个事件的间隔时间
        interval = exprnd(1/lambda);
        current_time = current_time + interval;
        
        if current_time < T
            event_times(end+1) = current_time;
        end
    end
end