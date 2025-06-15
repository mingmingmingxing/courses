function rayleigh_distribution_demo()
    % 参数设置
    sigma = 2;          % 瑞利分布参数
    n_samples = 10000;  % 生成的样本数量
    nbins = 50;         % 直方图分箱数
    
    % 使用反变换法生成瑞利分布随机变量
    uniform_samples = rand(n_samples, 1);  % 生成均匀分布随机数
    rayleigh_samples = sqrt(-2 * sigma^2 * log(1 - uniform_samples));  % 反变换
    
    % 计算理论概率密度函数
    r = linspace(0, max(rayleigh_samples), 1000);
    theoretical_pdf = (r / sigma^2) .* exp(-r.^2 / (2 * sigma^2));
    
    % 绘制结果
    figure;
    
    % 绘制样本直方图与理论PDF比较
    subplot(2,1,1);
    histogram(rayleigh_samples, nbins, 'Normalization', 'pdf');
    hold on;
    plot(r, theoretical_pdf, 'r-', 'LineWidth', 2);
    title('瑞利分布样本与理论PDF比较');
    xlabel('r');
    ylabel('概率密度');
    legend('样本直方图', '理论PDF');
    grid on;
    
    % 绘制经验CDF与理论CDF比较
    subplot(2,1,2);
    [f, x] = ecdf(rayleigh_samples);
    plot(x, f, 'b-');
    hold on;
    plot(r, 1 - exp(-r.^2 / (2 * sigma^2)), 'r-', 'LineWidth', 2);
    title('经验CDF与理论CDF比较');
    xlabel('r');
    ylabel('累积概率');
    legend('经验CDF', '理论CDF', 'Location', 'southeast');
    grid on;
    
    % 计算统计量
    sample_mean = mean(rayleigh_samples);
    sample_var = var(rayleigh_samples);
    
    % 理论统计量
    theoretical_mean = sigma * sqrt(pi/2);
    theoretical_var = (4 - pi)/2 * sigma^2;
    
    % 显示统计结果
    fprintf('瑞利分布参数 σ = %.2f\n', sigma);
    fprintf('样本数量: %d\n', n_samples);
    fprintf('样本均值: %.4f (理论: %.4f)\n', sample_mean, theoretical_mean);
    fprintf('样本方差: %.4f (理论: %.4f)\n', sample_var, theoretical_var);
    
    % Kolmogorov-Smirnov检验
    [h, p] = kstest(rayleigh_samples, 'CDF', [r', 1 - exp(-r.^2 / (2 * sigma^2))']);
    fprintf('\nK-S检验结果:\n');
    fprintf('h = %d (0表示接受原假设，1表示拒绝)\n', h);
    fprintf('p值 = %.4f\n', p);
    if h == 0
        fprintf('结论: 样本分布与理论瑞利分布无显著差异\n');
    else
        fprintf('结论: 样本分布与理论瑞利分布存在显著差异\n');
    end
end