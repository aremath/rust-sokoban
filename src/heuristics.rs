// Heuristics for Sokoban
// Includes basic lock detection / dead box detection
// Includes minimum matching heuristic for box distance from available targets
use pathfinding::matrix::Matrix;
use pathfinding::kuhn_munkres::kuhn_munkres_min;
use ordered_float::OrderedFloat;
use strum::IntoEnumIterator;
extern crate nalgebra as na;
use na::Vector2;
use std::cmp::{min, max};
use std::collections::HashMap;

use crate::sokoengine::{SokoManager, SokoState, HasVecs, MapTile, Entity, Direction, SokoInterface, Stringable};

// The documentation for the Kuhn-Munkres algorithm in pathfinding says:
// Also, using indefinite values such as positive or negative infinity or NaN can cause this function to loop endlessly.
// So I'm defining a BIG NUMBER for use in the minimum matching algorithm
// If the result of the minimum matching is larger than BIG_NUMBER, then the matching has failed
const BIG_NUMBER: isize = 1_000_000;

fn rotate_90(v: &Vector2<isize>) -> Vector2<isize> {
    return Vector2::new(-v.y, v.x);
}

fn rotate_180(v: &Vector2<isize>) -> Vector2<isize> {
    return Vector2::new(-v.x, -v.y);
}

fn rotate_270(v: &Vector2<isize>) -> Vector2<isize> {
    return Vector2::new(v.y, -v.x);
}

fn check_trap(s: &SokoState<MapTile, Entity>, p: &Vector2<isize>, dv: &Vector2<isize>, dv_p: &Vector2<isize>, d: Direction) -> Option<Trap> {
    let start = p.clone();
    let mut end = p.clone();
    while s.get_tile(&end) != MapTile::Wall {
        // The
        // WP->
        // ?_??
        // case: not a trap in the -> direction
        if s.get_tile(&(end + dv_p)) != MapTile::Wall {
            return None
        }
        end = end + dv;
    }
    // Subtract dv so that the code for finding the within doesn't depend on the direction
    // Without this, the trap is [start, end), but the side of the trap that end is on is dependent on the direction
    // With this, it's [start, end], so if your A-coord matches and your B-coord is within the range, then you are in it
    end = end - dv;
    match d {
        // Vertical Coordinate (.x) must match
        Direction::Left | Direction::Right => {
            assert!(start.x == end.x);
            let min = min(start.y, end.y);
            let max = max(start.y, end.y);
            Some(Trap { fixed_c: start.x, start: min, end: max, direction: d })
        }
        // Horizontal Coordinate (.y) must match
        Direction::Up | Direction::Down => {
            assert!(start.y == end.y);
            let min = min(start.x, end.x);
            let max = max(start.x, end.x);
            Some(Trap { fixed_c: start.y, start: min, end: max, direction: d })
        }

    }
}

#[derive(Debug, Hash, PartialEq, Clone)]
pub struct Trap {
    fixed_c: isize,
    start: isize,
    end: isize,
    direction: Direction,
}

impl Eq for Trap {}

//TODO: also include Corners!

// A corner is a structure of the form:
// WB
// ?W
// A box at B cannot be moved
// And therefore can only match with a target that it is already on top of
pub fn find_corners<T: HasVecs>(s: &SokoState<MapTile, Entity>, mgr: &T) -> Vec<Vector2<isize>> {
    let mut corners = Vec::new();
    for ((y, x), _) in s.map_layer.indexed_iter() {
        let iv = Vector2::new(y as isize, x as isize);
        // Corners start on a non-wall tile
        if s.get_tile(&iv) != MapTile::Wall {
            for d in Direction::iter() {
                let dv = mgr.d_to_v(d);
                let dv_p = rotate_90(dv);
                if s.get_tile(&(iv + dv)) == MapTile::Wall && s.get_tile(&(iv + dv_p)) == MapTile::Wall {
                    if !corners.contains(&iv) {
                        corners.push(iv);
                    }
                }
            }
        }
    }
    return corners;
}

// Computes the "adjacency" of boxes: which boxes can cover which targets
// Used to evaluate a SokoState
// Actually computing the answer to this is difficult, so I use a very simple method:
// Each block is assumed to be able to cover all targets, unless it is in a situation analagous to:
// W_B_O_W
// WWWWWWW
// If this situation exists, then that block can only cover that target (or other targets on the same wall)
// NOTE: this does not cover a more complex case like
// W___B__W
// WWWWW_WW
// WWWWWWWW
// Where the block is still trapped, but nontrivially so
//TODO: because the possible wall trap locations do not change, precompute them!
//TODO: Polymorphism over the M, E types
//  would require a trait for getting the appropriate Wall and Blank types
pub fn find_wall_traps<T: HasVecs>(s: &SokoState<MapTile, Entity>, mgr: &T) -> Vec<Trap> {
    let mut wall_traps = Vec::new();
    for ((y, x), _) in s.map_layer.indexed_iter() {
        let iv = Vector2::new(y as isize, x as isize);
        // The trap has to start on a non-wall tile
        if s.get_tile(&iv) != MapTile::Wall {
            // d is the direction of the wall trap
            for d in Direction::iter() {
                let dv = mgr.d_to_v(d);
                let dv_back = rotate_180(dv);
                let dv_p = rotate_90(dv);
                // It can be a trap if it looks like:
                // WP
                // ?W
                // Or any rotation
                // Only have to check +90' because the other direction will cover the +270' case
                // (Wall traps are always symmetrical)
                // Then check that there are no gaps in the wall
                // WP____W
                // ?WWWWW?
                // Or any rotation
                if s.get_tile(&(iv + dv_back)) == MapTile:: Wall {
                    match check_trap(s, &iv, &dv, &dv_p, d) {
                        Some(t) => { wall_traps.push(t); }
                        None => {}
                    }
                }
            }
        }
    }
    return wall_traps;
}

fn in_trap(p: &Vector2<isize>, trap: &Trap) -> bool {
    match trap.direction {
        // Vertical coordinate (which is .x) has to match
        Direction::Left | Direction::Right => {
            return p.x == trap.fixed_c && p.y >= trap.start && p.y <= trap.end;
        }
        // Horizontal coordinate (.y) has to match
        Direction::Up | Direction::Down => {
            return p.y == trap.fixed_c && p.x >= trap.start && p.x <= trap.end;
        }
    }
}

// Struct to hold all the important heuristic info that only has to be computed once
pub struct HeuristicHelper {
    target_locs: Vec<Vector2<isize>>,
    wall_traps: Vec<Trap>,
    trap_to_targets: HashMap<Trap, Vec<Vector2<isize>>>,
    corners: Vec<Vector2<isize>>,
    //corners_to_targets: HashMap<Vector2<isize>, Vec<Vector2<isize>>,
}

impl HeuristicHelper {

    pub fn new<T: HasVecs>(s: &SokoState<MapTile, Entity>, mgr: &T) -> Self {
        let mut target_locs = Vec::new();
        for ((y, x), _) in s.map_layer.indexed_iter() {
            let iv = Vector2::new(y as isize, x as isize);
            if s.get_tile(&iv) == MapTile::Target {
                target_locs.push(iv.clone());
            }
        }
        let traps = find_wall_traps(s, mgr);
        let mut traps_to_targets = HashMap::new();
        for t in &traps {
            traps_to_targets.insert(t.clone(), Vec::new());
        }
        for loc in &target_locs {
            for t in &traps {
                if in_trap(loc, t) {
                    traps_to_targets.get_mut(t).expect("Couldn't find the trap!").push(loc.clone());
                }
            }
        }
        let corners = find_corners(s, mgr);
        return HeuristicHelper { target_locs: target_locs, wall_traps: traps,
            trap_to_targets: traps_to_targets,
            corners: corners };
    }
}

// Yes, manhattan distance is non-negative
// No, that doesn't mean I'm going to use usize
fn manhattan(v1: &Vector2<isize>, v2: &Vector2<isize>) -> isize {
    return (v1.x - v2.x).abs() + (v1.y - v2.y).abs();
}

// Find the block adjacency matrix, which associates targets to possible blocks
// Trapped blocks can only be associated with targets inside their respective traps
// Also does a bunch of other stuff, as finding the block locs and iterating them is done once
pub fn find_box_adj(s: &SokoState<MapTile, Entity>, helper: &HeuristicHelper) -> (Matrix<isize>, isize, Option<isize>) {
    let player_loc = &s.player_locs[0]; // This assumes both that there IS a player_loc,
    // And that there's not more than one player
    let mut block_locs = Vec::new();
    for ((y, x), _) in s.map_layer.indexed_iter() {
        let iv = Vector2::new(y as isize, x as isize);
        if s.get_entity(&iv) == Entity::Block {
            block_locs.push(iv);
        }
    }
    // Compute the distance table
    let mut distance_matrix = Matrix::new(helper.target_locs.len(), block_locs.len(), 0);
    let mut n_satisfied = 0;
    let mut player_to_box = None;
    let mut sum = 0;
    // Now that we have the locs, find out which targets they belong to
    // Each block is in at most two traps
    // If a block is in two traps, that means it's in their shared corner, and can't be moved at all
    // Therefore can only occupy a target that it's on
    for (j, block) in (&block_locs).into_iter().enumerate() {
        let mut block_on = false;
        // Count the number of boxes that are actually ON targets
        //TODO: fractional? I don't think it makes sense to dilute the heuristic for longer levels though...
        for target in &helper.target_locs {
            if block == target {
                n_satisfied += 1;
                block_on = true;
            }
        }
        // Compute the distance from the player to all non-satisfied blocks
        if !block_on {
            match player_to_box {
                Some(m) => {
                    let d = manhattan(player_loc, block);
                    if d < m {
                        player_to_box = Some(d);
                    }
                }
                None => {
                    player_to_box = Some(manhattan(player_loc, block));
                }
            }
        }
        // Compute all the target stuff
        let mut n_traps = 0;
        for (trap, targets) in &helper.trap_to_targets {
            // block j is in the trap and can only be assigned to one of that trap's targets
            if in_trap(block, trap) {
                //println!("{}, {:?}", block, trap);
                //println!("{}, {}", block.x, block.y);
                n_traps += 1;
                for (i, target) in (&helper.target_locs).into_iter().enumerate() {
                    if targets.contains(target) {
                        let m = manhattan(target, block);
                        distance_matrix[(i, j)] = manhattan(target, block);
                        sum += m;
                    // Other targets are unreachable
                    } else {
                        distance_matrix[(i, j)] = BIG_NUMBER;
                    }
                }
            }
        }
        match n_traps {
            // If the block is in zero traps, then it can believably reach any target
            0 => {
                for (i, target) in (&helper.target_locs).into_iter().enumerate() {
                    let m = manhattan(target, block);
                    distance_matrix[(i, j)] = manhattan(target, block);
                    sum += m;
                }
            }
            // If the block is in one trap, no need to do anything since the distances have already been set
            1 => {}
            // If the block is in two traps, it is in a corner and cannot be moved at all
            2 => {
                for (i, target) in (&helper.target_locs).into_iter().enumerate() {
                    if target != block {
                        sum -= distance_matrix[(i, j)];
                        distance_matrix[(i, j)] = BIG_NUMBER;
                    }
                }
            }
            _ => { panic!("Block somehow in more than 2 traps!"); }
        }
        // Not all corners are made from wall traps
        // This handles other corners
        for corner in &helper.corners {
            if block == corner {
                for (i, target) in (&helper.target_locs).into_iter().enumerate() {
                    if target == corner {
                        // distance_matrix[(i, j)] Should be 0 at this point
                        sum -= distance_matrix[(i, j)];
                        distance_matrix[(i, j)] = 0;
                    // Other targets are unreachable
                    } else {
                        sum -= distance_matrix[(i, j)];
                        distance_matrix[(i, j)] = BIG_NUMBER;
                    }
                }
            }
        }
    }
    //println!("{:?}", block_locs);
    //println!("{:?}", helper.target_locs);
    //println!("{:?}", distance_matrix);
    assert!(sum < BIG_NUMBER, "Distances are too large!");
    return (distance_matrix, n_satisfied, player_to_box);
}

/*
pub fn simple_heuristic(s: &SokoState<MapTile, Entity>) -> OrderedFloat<f64> {
    return OrderedFloat(0.0); //TODO
}
*/

//TODO: Some combination of total box distance and median box distance?
pub fn matching_heuristic(s: &SokoState<MapTile, Entity>, helper: &HeuristicHelper) -> OrderedFloat<f64> {
    let (m, _, _) = find_box_adj(s, helper);
    let (c, _) = kuhn_munkres_min(&m);
    // This means that a block was matched with an infeasible target
    // In turn, that means there is a target with no feasible blocks
    // In that case, the state is effectively dead.
    if c >= BIG_NUMBER {
        return OrderedFloat(f64::INFINITY);
    }
    return OrderedFloat(c as f64);
}

// Matching heuristic, but you get bonus points for having boxes on targets, winning, and being near boxes
pub fn matching_heuristic_with_extras(s: &SokoState<MapTile, Entity>, helper: &HeuristicHelper) -> OrderedFloat<f64> {
    let (matrix, n_sat, p) = find_box_adj(s, helper);
    let (matching, _) = kuhn_munkres_min(&matrix);
    // You get nothing if the matching is impossible
    if matching >= BIG_NUMBER {
        return OrderedFloat(0.0);
    }
    let player_d = match p {
        Some(d) => { d }
        // There are no unmatched blocks
        // Usually, this means victory, but I won't assign a penalty in this case
        None => { 0 }
    };
    let mut victory = OrderedFloat(0.0);
    if s.is_win() {
        victory = OrderedFloat(100.0);
    }
    let n_targets = helper.target_locs.len() as isize;
    let n_unsat = n_targets - n_sat;
    let box_penalty = OrderedFloat(n_unsat as f64) * OrderedFloat(10.0);
    // The player going towards an unsatisfied block is 1/3 as important as actually making progress on the blocks
    //TODO: this gets weird if the player pushes a block onto a target and the next one is very far away
    //ALSO: the player may have to still push a matched block
    let player_penalty = OrderedFloat(player_d as f64 / 3.0);
    return (OrderedFloat(1.0) + victory) / (OrderedFloat(matching as f64) + box_penalty + player_penalty + 0.01);
}

//TODO: player position near a box heuristic for ordering
pub fn matching_heuristic_inv(s: &SokoState<MapTile, Entity>, helper: &HeuristicHelper) -> OrderedFloat<f64> {
    let mut victory = OrderedFloat(0.0);
    if s.is_win() {
        victory = OrderedFloat(100.0);
    }
    // For matching_heuristic, smaller is better
    // For this heuristic, bigger is better
    // Use 1 / x + 0.01 so that if matching_heuristic == 0 (for example, if the game is won), the value isn't infty
    // This means max value is 100.0
    // This is arbitrary
    return (OrderedFloat(1.0) + victory) / (matching_heuristic(s, helper) + 0.01);
}
