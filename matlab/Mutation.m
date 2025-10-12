function x_new = Mutation(x, dpp, D)
    [n_candidate, n_types] = size(x);
    built_idx = find(any(x,2));  % rows with centers
    empty_idx = find(~any(x,2));  % empty locations

    % Try to swap one existing center with a new one
    success = false;
    attempt = 0;
    while ~success && attempt < 500
        i_old = built_idx(randi(length(built_idx)));
        t_old = find(x(i_old,:) == 1);

        i_new = empty_idx(randi(length(empty_idx)));
        % D constraint
        conflict = false;
        for j = 1:n_candidate
            if j ~= i_old && any(x(j,:)) && dpp(i_new, j) < D
                conflict = true;
                break;
            end
        end

        if ~conflict
            x_new = x;
            x_new(i_old, :) = 0;
            x_new(i_new, randi(n_types)) = 1;
            success = true;
            return;
        end

        attempt = attempt + 1;
    end

    % If mutation failed, return unchanged
    x_new = x;
end
