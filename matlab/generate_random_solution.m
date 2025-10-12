function ind = generate_random_solution(n_candidate, n_community, Q, D, dpp, d, E_L, E_U, U, C, Cp, Cpp, gama)

    n_types = length(Q);
    
    % Step 1: Randomly select U valid locations with distance constraints
    x = zeros(n_candidate, n_types);
    selected = randperm(n_candidate);
    count = 0;
    
    for idx = 1:n_candidate
        if count >= U
            break;
        end
        loc = selected(idx);
        
        % Check D constraint
        too_close = false;
        for j = 1:n_candidate
            if any(x(j,:)) && dpp(loc, j) < D
                too_close = true;
                break;
            end
        end
        if too_close
            continue;
        end
        
        % Assign a random type
        t = randi(n_types);
        x(loc, t) = 1;
        count = count + 1;
    end

    % Step 2: Generate fuzzy demand
    E = floor(((E_U - E_L) .* rand(n_community, 1)) + E_L);

    % Step 3: Allocate demands
    y = allocation(x, E, Q, d);

    % Step 4: Calculate cost
    cost = OF(x, y, E, C, Cp, Cpp, d, gama);

    % Create output individual structure
    ind.x = x;
    ind.E = E;
    ind.y = y;
    ind.Cost = cost;
end
