function mt_uniform_test()
    % 参数设置
    n = 1e6;  % 生成的随机数数量
    
    % 生成(0,1)区间上的均匀分布随机数
    rng('default');  % 使用Mersenne Twister算法，默认种子
    uniform_rand = rand(n, 1);
    
    % 计算统计量
    sample_mean = mean(uniform_rand);
    sample_var = var(uniform_rand);
    
    % 理论值
    theoretical_mean = 0.5;
    theoretical_var = 1/12;  % 均匀分布U(0,1)的方差
    
    % 显示统计结果
    fprintf('Mersenne Twister生成 %d 个随机数的统计结果:\n', n);
    fprintf('样本均值: %.6f (理论: 0.5)\n', sample_mean);
    fprintf('样本方差: %.6f (理论: 0.083333)\n', sample_var);
    
    % 绘制概率密度图
    figure;
    subplot(2,1,1);
    histogram(uniform_rand, 100, 'Normalization', 'pdf');
    hold on;
    plot([0 1], [1 1], 'r-', 'LineWidth', 2);
    title('Mersenne Twister随机数概率密度分布');
    xlabel('值');
    ylabel('概率密度');
    legend('样本分布', '理论均匀分布');
    grid on;
    
    % 自相关检验
    figure;
    autocorr(uniform_rand, 50);
    title('Mersenne Twister随机数自相关函数');
end