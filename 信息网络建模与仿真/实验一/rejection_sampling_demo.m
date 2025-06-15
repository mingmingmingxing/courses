function acceptance_rejection_demo()
    clc;
    close all;
    
    % 参数设置
    n_samples = 10000;      % 生成样本数
    nbins = 50;             % 直方图分箱数
    fprintf('Generating %d samples...\n', n_samples);
    
    % 方法1：矩形区域舍选法
    tic;
    [samples1, attempts1] = method1_dynamic(n_samples);
    time1 = toc;
    actual_eff1 = n_samples / attempts1;
    
    % 方法2：分段线性参考分布
    tic;
    [samples2, attempts2] = method2_dynamic(n_samples);
    time2 = toc;
    actual_eff2 = n_samples / attempts2;
    
    % 理论效率计算
    C1 = 135 / 64;          % 方法1的理论常数
    C2 = 1.3167;            % 方法2的理论常数
    theoretical_eff1 = 1 / C1;
    theoretical_eff2 = 1 / C2;
    
    figure('Position', [100, 100, 1200, 500]);
    
    % 方法1结果
    subplot(1, 2, 1);
    % 绘制直方图（归一化为PDF）
    h1 = histogram(samples1, nbins, 'Normalization', 'pdf', ...
                  'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
    hold on;
    % 理论PDF曲线
    x = linspace(0, 1, 500);
    plot(x, 20 * x .* (1 - x).^3, 'r-', 'LineWidth', 2);
    % 图表修饰
    title(sprintf('Method 1: Rectangular (Eff=%.1f%%)', actual_eff1*100));
    xlabel('x');
    ylabel('Probability Density');
    legend('Generated Samples', 'Theoretical PDF', 'Location', 'northwest');
    grid on;
    xlim([0 1]);
    ylim([0 2.5]);
    
    % 方法2结果
    subplot(1, 2, 2);
    % 绘制直方图
    h2 = histogram(samples2, nbins, 'Normalization', 'pdf', ...
                  'FaceColor', [0.4 0.8 0.4], 'EdgeColor', 'none');
    hold on;
    % 理论PDF曲线
    plot(x, 20 * x .* (1 - x).^3, 'r-', 'LineWidth', 2);
    % 图表修饰
    title(sprintf('Method 2: Piecewise Linear (Eff=%.1f%%)', actual_eff2*100));
    xlabel('x');
    ylabel('Probability Density');
    legend('Generated Samples', 'Theoretical PDF', 'Location', 'northwest');
    grid on;
    xlim([0 1]);
    ylim([0 2.5]);
    
    % 性能比较输出
    fprintf('\n=== Performance Comparison ===\n');
    fprintf('Method 1 (Rectangular):\n');
    fprintf('  Theoretical Efficiency: %.2f%%\n', theoretical_eff1 * 100);
    fprintf('  Actual Efficiency:    %.2f%%\n', actual_eff1 * 100);
    fprintf('  Execution Time:       %.4f sec\n', time1);
    
    fprintf('\nMethod 2 (Piecewise Linear):\n');
    fprintf('  Theoretical Efficiency: %.2f%%\n', theoretical_eff2 * 100);
    fprintf('  Actual Efficiency:    %.2f%%\n', actual_eff2 * 100);
    fprintf('  Execution Time:       %.4f sec\n', time2);
    
    fprintf('\nEfficiency Improvement: %.1f%%\n', ...
        (actual_eff2 - actual_eff1)/actual_eff1 * 100);
end

% 子函数实现
function [samples, attempts] = method1_dynamic(n_samples)
    % 方法1：矩形区域舍选法
    samples = zeros(1, n_samples);
    count = 0;
    attempts = 0;
    C = 135 / 64;  % max(f(x)) = 135/64 at x=1/4
    
    while count < n_samples
        u1 = rand();
        u2 = rand();
        attempts = attempts + 1;
        if u1 <= 20 * u2 * (1 - u2)^3 / C
            count = count + 1;
            samples(count) = u2;
        end
    end
end

function [samples, attempts] = method2_dynamic(n_samples)
    % 方法2：分段线性参考分布
    segments = [
        struct('a', 0.0,  'b', 0.11, 't', @(x) 21*x,   'inv_cdf', @(u) sqrt(u/10.5), 'prob', 0.0965);
        struct('a', 0.11, 'b', 0.25, 't', @(x) 2.31,   'inv_cdf', @(u) 0.11 + u*0.14, 'prob', 0.2456);
        struct('a', 0.25, 'b', 1.0,  't', @(x) -3.08*x + 3.08, ...
               'inv_cdf', @(u) 1 - sqrt(0.9375 - u*0.8625), 'prob', 0.6579)
    ];
    
    samples = zeros(1, n_samples);
    count = 0;
    attempts = 0;
    
    while count < n_samples
        % 选择分段
        u_seg = rand();
        cum_probs = cumsum([segments.prob]);
        seg_idx = find(u_seg <= cum_probs, 1);
        seg = segments(seg_idx);
        
        % 生成候选样本
        u2 = seg.inv_cdf(rand());
        attempts = attempts + 1;
        
        % 接受/拒绝
        if rand() <= 20 * u2 * (1 - u2)^3 / seg.t(u2)
            count = count + 1;
            samples(count) = u2;
        end
    end
end