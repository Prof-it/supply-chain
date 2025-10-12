% Extract objectives
obj1 = arrayfun(@(s) s.Cost(1), F1);
obj2 = arrayfun(@(s) s.Cost(2), F1);

% Identify extremes
[~, idx_min_obj1] = min(obj1);  % Best in objective 1
[~, idx_min_obj2] = min(obj2);  % Best in objective 2

% Plot all points
figure;
scatter(obj1, obj2, 60, 'filled', 'MarkerFaceColor', [0.6 0.6 0.6]); hold on;

% Highlight extremes
scatter(obj1(idx_min_obj1), obj2(idx_min_obj1), 100, 'r', 'filled');  % Red = min obj1
scatter(obj1(idx_min_obj2), obj2(idx_min_obj2), 100, 'b', 'filled');  % Blue = min obj2

xlabel('Objective 1 (Cost)');
ylabel('Objective 2 (Standrad Deviation)');
legend('Pareto Solutions','Min Objective 1','Min Objective 2');
title('Pareto Front with Extreme Points');
grid on;





selected = F1(9);
candidates = xy_candidate;
% SELECTED LOCATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;

% Plot all candidate locations in gray
h_all = scatter(candidates(:,1), candidates(:,2), 40, [0.6 0.6 0.6], 'filled');

% Extract selected facilities and their types
[selected_ids, facility_types] = find(selected.x);

% Define colors for types (Capacity 30 = red, 40 = green, 50 = blue)
type_colors = [1 0 0; 0 0.5 0; 0 0 1];

% Store scatter handles for legend
h_types = gobjects(3,1);

% Plot selected facilities by type
for t = 1:3
    idx = facility_types == t;
    h_types(t) = scatter(candidates(selected_ids(idx),1), ...
                         candidates(selected_ids(idx),2), ...
                         120, type_colors(t,:), 'filled', 'MarkerEdgeColor', 'k');
end

xlabel('X Coordinate');
ylabel('Y Coordinate');
title('Selected Facility Locations by Capacity Type');
legend([h_all, h_types'], {'All Candidates', 'Capacity 30', 'Capacity 40', 'Capacity 50'});
grid on;


% ALLOCATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
communities = xy_community;

% Find all nonzero allocations
[i_fac, j_com, flows] = find(selected.y);  % facility i, community j

% Create rows with: Community ID, Facility ID, Allocation
n_rows = length(flows);
CommunityID = j_com(:);
FacilityID = i_fac(:);
AllocatedShare = flows(:);


%%%%%%%%%%%%%%%%table%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create table
T = table(CommunityID, FacilityID, flows, ...
          'VariableNames', {'CommunityID', 'FacilityID', 'AllocatedDemand'});


% Display top of table
disp(head(T, 27));  % show first 10 rows

% Optional: sort by community or allocation share
T = sortrows(T, {'CommunityID', 'AllocatedDemand'}, {'ascend', 'descend'});

% Export to Excel or CSV if needed
% writetable(T, 'allocations.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find selected locations and their types
[selected_locs, nh_types] = find(selected.x);  % location index, type index
n_selected = length(selected_locs);

% Initialize arrays
Location = selected_locs;
Type = nh_types;
Capacity = Q(nh_types);  % Q = array of capacities by type

AllocatedCommunities = strings(n_selected, 1);
AllocatedPeople = strings(n_selected, 1);

for k = 1:n_selected
    i = selected_locs(k);  % facility index
    assigned_coms = find(selected.y(i,:) > 0);  % assigned communities
    assigned_vals = selected.y(i, assigned_coms);  % allocated demand values

    % Store as comma-separated strings
    AllocatedCommunities(k) = strjoin(string(assigned_coms), ',');
    AllocatedPeople(k) = strjoin(string(round(assigned_vals, 2)), ',');
end

% Create the table
T = table(Location, Type, Capacity(:), AllocatedCommunities, AllocatedPeople, ...
    'VariableNames', {'Location', 'The_nursing_home_type', 'Capacity', ...
                      'Allocated_communities', 'The_number_of_allocated_people'});

% Display a preview
disp(T);

% Optionally export to Excel or CSV
% writetable(T, 'location_allocation_summary.csv');
% or for Word-compatible format:
writetable(T, 'location_allocation_summary2.txt', 'Delimiter', '\t');


