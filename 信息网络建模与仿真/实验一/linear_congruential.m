function lcg_analysis()
    % 参数设置
    n = 1e6;            % 生成的随机数数量
    seed = 12345;       % 初始种子
    a = 16807;          % 乘数
    m = 2^31 - 1;       % 模数 (2147483647)
    
    % 生成随机数序列
    rng_sequence = generate_lcg_sequence(n, seed, a, m);
    
    % 转换为(0,1)区间
    uniform_sequence = rng_sequence / m;
    
    % 计算统计量
    sample_mean = mean(uniform_sequence);
    sample_var = var(uniform_sequence);
    
    % 理论值
    theoretical_mean = 0.5;
    theoretical_var = 1/12;
    
    % 显示统计结果
    fprintf('线性同余法生成 %d 个随机数的统计结果:\n', n);
    fprintf('样本均值: %.6f (理论: 0.5)\n', sample_mean);
    fprintf('样本方差: %.6f (理论: 0.083333)\n', sample_var);
    
    % 绘制概率密度图
    figure;
    subplot(2,1,1);
    histogram(uniform_sequence, 100, 'Normalization', 'pdf');
    hold on;
    plot([0 1], [1 1], 'r-', 'LineWidth', 2);
    title('随机数概率密度分布');
    xlabel('值');
    ylabel('概率密度');
    legend('样本分布', '理论均匀分布');
    grid on;
    
end

function sequence = generate_lcg_sequence(n, seed, a, m)
    % 线性同余法生成随机数序列
    sequence = zeros(n, 1);
    sequence(1) = seed;
    
    for i = 2:n
        sequence(i) = mod(a * sequence(i-1), m);
    end
end