function mixed_gaussian_demo()
    % 参数设置
    n_samples = 100000;  % 生成的样本数量
    
    % 定义混合高斯分布的参数
    p = [1/2, 1/3, 1/6];    % 各成分的权重
    a = [-1, 0, 1];          % 各成分的均值
    b = [1/4, 1, 1/2];       % 各成分的标准差
    
    % 确保权重总和为1
    assert(abs(sum(p) - 1) < 1e-10, '权重总和必须为1');
    
    % 生成混合高斯随机变量
    samples = generate_mixed_gaussian(n_samples, p, a, b);
    
    % 绘制概率密度图
    figure;
    
    % 绘制样本直方图
    histogram(samples, 200, 'Normalization', 'pdf');
    hold on;
    
    % 计算并绘制理论概率密度曲线
    x = linspace(-3, 3, 1000);
    theoretical_pdf = zeros(size(x));
    
    for i = 1:length(p)
        theoretical_pdf = theoretical_pdf + p(i) * normpdf(x, a(i), b(i));
    end
    
    plot(x, theoretical_pdf, 'r-', 'LineWidth', 2);
    
    % 绘制各成分的高斯分布
    colors = {'g--', 'm--', 'c--'};
    for i = 1:length(p)
        plot(x, p(i) * normpdf(x, a(i), b(i)), colors{i}, 'LineWidth', 1.5);
    end
    
    % 图表装饰
    title('混合高斯分布的概率密度函数');
    xlabel('x');
    ylabel('概率密度');
    legend('样本直方图', '理论混合分布', ...
           '成分1: N(-1,1/4)', '成分2: N(0,1)', '成分3: N(1,1/2)');
    grid on;
    
    % 显示统计信息
    fprintf('生成的样本数量: %d\n', n_samples);
    fprintf('样本均值: %.4f\n', mean(samples));
    fprintf('样本方差: %.4f\n', var(samples));
end

function samples = generate_mixed_gaussian(n, p, a, b)
    % 使用组合法生成混合高斯随机变量
    % 输入:
    %   n - 样本数量
    %   p - 各成分的权重向量
    %   a - 各成分的均值向量
    %   b - 各成分的标准差向量
    % 输出:
    %   samples - 生成的混合高斯随机变量
    
    % 生成选择成分的随机数
    component = randsample(length(p), n, true, p);
    
    % 预分配样本数组
    samples = zeros(n, 1);
    
    % 生成各成分的高斯随机变量
    for i = 1:length(p)
        mask = (component == i);
        n_i = sum(mask);
        
        % 使用Box-Muller方法生成高斯随机变量
        [z1, z2] = box_muller(ceil(n_i/2));
        z = [z1; z2];
        z = z(1:n_i);  % 确保数量正确
        
        % 转换为指定均值和标准差
        samples(mask) = a(i) + b(i) * z;
    end
end

function [x, y] = box_muller(n)
    % Box-Muller方法生成标准高斯随机变量
    u1 = rand(n, 1);
    u2 = rand(n, 1);
    
    r = sqrt(-2 * log(u1));
    theta = 2 * pi * u2;
    
    x = r .* cos(theta);
    y = r .* sin(theta);
end