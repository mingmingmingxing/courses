function test_MRG32k3a()
    % 参数设置
    n = 10000;  % 生成随机数的数量
    seed = [12345, 12345, 12345, 12345, 12345, 12345];  % 种子
    
    % 生成随机数
    rand_nums = MRG32k3a_generate(n, seed);
    
    % 计算统计量
    sample_mean = mean(rand_nums);
    sample_var = var(rand_nums);
    
    % 理论值
    theoretical_mean = 0.5;
    theoretical_var = 1/12;  % 均匀分布U(0,1)的方差
    
    % 显示结果
    fprintf('生成的随机数数量: %d\n', n);
    fprintf('样本均值: %.6f (理论: 0.5)\n', sample_mean);
    fprintf('样本方差: %.6f (理论: 0.083333)\n', sample_var);
    
    % 绘制直方图
    figure;
    histogram(rand_nums, 100, 'Normalization', 'pdf');
    hold on;
    plot([0 1], [1 1], 'r-', 'LineWidth', 2);  % 理论均匀分布
    title('随机数分布 vs 理论均匀分布');
    xlabel('值');
    ylabel('概率密度');
    legend('生成的随机数', '理论均匀分布');
end

function rand_nums = MRG32k3a_generate(n, seed)
    % MRG32k3a随机数生成器
    % 输入:
    %   n - 要生成的随机数数量
    %   seed - 6个元素的种子向量 [x11, x12, x13, x21, x22, x23]
    % 输出:
    %   rand_nums - 生成的随机数向量(n×1)
    
    % 定义常数
    m1 = 2^32 - 209;
    m2 = 2^32 - 22853;
    
    % 初始化状态
    x1 = mod(seed(1:3), m1);  % 第一组状态
    x2 = mod(seed(4:6), m2);  % 第二组状态
    
    % 预分配输出数组
    rand_nums = zeros(n, 1);
    
    for i = 1:n
        % 更新第一组状态
        new_x1 = mod(1403580 * x1(2) - 810728 * x1(3), m1);
        x1 = [new_x1, x1(1:2)];  % 移位
        
        % 更新第二组状态
        new_x2 = mod(527612 * x2(1) - 1370589 * x2(3), m2);
        x2 = [new_x2, x2(1:2)];  % 移位
        
        % 组合生成随机数
        if new_x1 <= new_x2
            rand_nums(i) = (new_x1 - new_x2 + m1) / (m1 + 1);
        else
            rand_nums(i) = (new_x1 - new_x2) / (m1 + 1);
        end
    end
end