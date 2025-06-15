function simple_fsr_test()
    % 参数设置
    n_samples = 10000;      % 生成样本量
    initial_state = 0x8A5D; % 推荐初始状态
    bit_length = 16;
    
    % 1. FSR核心实现
    function state = advance_fsr(state)
        feedback = bitxor(bitget(state,16), bitxor(bitget(state,12), ...
                  bitxor(bitget(state,3), bitget(state,1))));
        state = bitshift(state, -1);
        state = bitset(state, 16, feedback);
    end

    % 2. 生成随机数
    state = initial_state;
    samples = zeros(n_samples, 1);
    for i = 1:n_samples
        % 采集16位并跳过15个状态
        num = 0;
        for j = 1:16
            num = bitshift(num,1) + bitget(state,1);
            state = advance_fsr(state);
        end
        for k = 1:15
            state = advance_fsr(state);
        end
        samples(i) = num;
    end
    normalized = double(samples) / (2^bit_length);
    
    % 3. 基础统计验证
    fprintf('=== 基础统计 ===\n');
    fprintf('均值: %.6f (理论: 0.5)\n', mean(normalized));
    fprintf('方差: %.6f (理论: 0.0833)\n', var(normalized));
    
    % 4. 纯图形化验证
    figure('Position',[100,100,800,400]);
    
    % 4.1 概率密度分布
    subplot(1,2,1);
    histogram(normalized, 20, 'Normalization','pdf');
    hold on; plot([0 1],[1 1], 'r-', 'LineWidth',1.5);
    title('概率密度分布');
    xlabel('值'); ylabel('密度');
    legend('样本','理论');
    
    % 4.2 分位数-分位数图(Q-Q Plot)
    subplot(1,2,2);
    qqplot(normalized);
    title('Q-Q图验证均匀性');
    
    % 5. 自相关快速检查（仅打印前5阶）
    fprintf('\n自相关(滞后1-5): ');
    fprintf('%.3f  ', autocorr(normalized,5));
    fprintf('\n');
end