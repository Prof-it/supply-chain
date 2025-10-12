function [x_new, notpossible] = Crossover(p1, p2, Q, D, dpp, E_L, E_U)
    n_candidate = size(p1.x,1);
    n_types = size(p1.x,2);
    U = sum(p1.x(:));  % assumes same U for both parents

    notpossible = 0;
    success_flag = false;

    % Start with minimum (intersection) of parents
    x = min(p1.x, p2.x);
    total_built = sum(x(:));

    % List available locations
    remaining = find(sum(x,2) == 0);  % not yet assigned
    attempt = 0;

    while total_built < U && ~isempty(remaining) && attempt < 500
        i = remaining(randi(length(remaining)));
        % Check D constraint
        conflict = false;
        for j = 1:n_candidate
            if any(x(j,:)) && dpp(i,j) < D
                conflict = true;
                break;
            end
        end
        if ~conflict
            t = randi(n_types);
            x(i,t) = 1;
            total_built = total_built + 1;
        end
        remaining(remaining == i) = [];
        attempt = attempt + 1;
    end

    if total_built < U
        notpossible = 1;
        x_new = x;
        return;
    end

    x_new = x;
end
