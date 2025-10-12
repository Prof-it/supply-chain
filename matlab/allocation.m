function y = allocation(x, E, Q, d)
[n_candidate, n_community] = size(d);
n_types = size(x,2);
y = zeros(n_candidate, n_community);
remaining_E = E;

% Precompute capacity per built center
cap = zeros(n_candidate,1);
for i = 1:n_candidate
    for t = 1:n_types
        if x(i,t) == 1
            cap(i) = Q(t);
        end
    end
end

% Assign each community demand randomly
for j = 1:n_community
    comm_demand = remaining_E(j);
    while comm_demand > 0
        % Find available centers
        available = find(cap > 0);
        if isempty(available)
            break;
        end
        i = available(randi(length(available)));  % randomly pick one
        assign = min(comm_demand, cap(i));
        y(i,j) = y(i,j) + assign;
        cap(i) = cap(i) - assign;
        comm_demand = comm_demand - assign;
    end
end
end
