function gaussian_generation_comparison()
    % 参数设置
    n_samples = 1e6;  % 生成的样本数量
    
    % 方法1: Box-Muller方法
    tic;
    [x1, y1] = box_muller(n_samples);
    time1 = toc;
    
    % 方法2: 极坐标法
    tic;
    [x2, y2, efficiency] = polar_method(n_samples);
    time2 = toc;
    
    % 显示结果
    fprintf('生成 %d 对高斯随机变量的时间比较:\n', n_samples);
    fprintf('Box-Muller方法: %.4f 秒\n', time1);
    fprintf('极坐标法: %.4f 秒\n', time2);
    fprintf('极坐标法的舍选效率: %.2f%%\n', efficiency * 100);
    
    % 绘制分布
    figure;
    
    % Box-Muller方法结果
    subplot(2,2,1);
    histogram(x1, 100, 'Normalization', 'pdf');
    hold on;
    plot_gaussian_pdf();
    title('Box-Muller方法 X分布');
    xlabel('值');
    ylabel('概率密度');
    
    subplot(2,2,2);
    histogram(y1, 100, 'Normalization', 'pdf');
    hold on;
    plot_gaussian_pdf();
    title('Box-Muller方法 Y分布');
    xlabel('值');
    ylabel('概率密度');
    
    % 极坐标法结果
    subplot(2,2,3);
    histogram(x2, 100, 'Normalization', 'pdf');
    hold on;
    plot_gaussian_pdf();
    title('极坐标法 X分布');
    xlabel('值');
    ylabel('概率密度');
    
    subplot(2,2,4);
    histogram(y2, 100, 'Normalization', 'pdf');
    hold on;
    plot_gaussian_pdf();
    title('极坐标法 Y分布');
    xlabel('值');
    ylabel('概率密度');
    
    % 绘制X-Y散点图
    figure;
    subplot(1,2,1);
    scatter(x1, y1, '.');
    axis equal;
    title('Box-Muller方法 X-Y分布');
    xlabel('X');
    ylabel('Y');
    
    subplot(1,2,2);
    scatter(x2, y2, '.');
    axis equal;
    title('极坐标法 X-Y分布');
    xlabel('X');
    ylabel('Y');
end

function [x, y] = box_muller(n)
    % Box-Muller方法生成高斯随机变量
    u1 = rand(n, 1);
    u2 = rand(n, 1);
    
    % 转换公式
    r = sqrt(-2 * log(u1));
    theta = 2 * pi * u2;
    
    x = r .* cos(theta);
    y = r .* sin(theta);
end

function [x, y, efficiency] = polar_method(n)
    % 极坐标法生成高斯随机变量
    x = zeros(n, 1);
    y = zeros(n, 1);
    accepted = 0;
    total = 0;
    
    while accepted < n
        total = total + 1;
        
        % 生成均匀随机点
        u1 = 2 * rand() - 1;  % [-1,1]
        u2 = 2 * rand() - 1;  % [-1,1]
        s = u1^2 + u2^2;
        
        % 接受条件
        if s > 0 && s < 1
            accepted = accepted + 1;
            % 转换公式
            factor = sqrt(-2 * log(s) / s);
            x(accepted) = u1 * factor;
            y(accepted) = u2 * factor;
        end
    end
    
    efficiency = n / total;
end

function plot_gaussian_pdf()
    % 绘制标准高斯分布PDF
    x = linspace(-5, 5, 1000);
    y = 1/sqrt(2*pi) * exp(-x.^2/2);
    plot(x, y, 'r-', 'LineWidth', 1.5);
    legend('样本分布', '理论N(0,1)');
end