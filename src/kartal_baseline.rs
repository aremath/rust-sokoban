// Implementation of "Data-driven Sokoban Puzzle Generation with MCTS" by Kartal et al. (AIIDE 2016)

// Initial state: SokoState<MapTile, Entity>
//

extern crate nalgebra as na;
use na::Vector2;
use ndarray::{Array, Ix2};
use strum::IntoEnumIterator;
use ordered_float::OrderedFloat;
use std::cmp;
use std::iter::zip;

use crate::sokoengine::{SokoState, Direction, MapTile, Entity, SokoManager, HasVecs, Stringable};
use crate::mcts::{Searchable, TaggedSokoState};

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub enum GenStage {
    LevelGen,
    SolGen,
    Eval
}

//TODO: would be nice to have the following:
// update(action) function that takes an action and returns a state s.t.
// the neighbors() function is consistent with update
// This would require uniqueness of the actions (e.g. AddBox(x,y))
pub enum GenAction {
    AddBox,
    DelObstacle,
    Freeze,
    MovePlayer,
    Evaluate,
}

type BoxMapping = Vec<(Vector2<isize>, Vector2<isize>)>;

// level_final -> TaggedSokoState in order to enforce DAGness
#[derive(Clone, Hash, PartialEq)]
pub struct GenState {
    pub stage: GenStage,
    pub level_init: SokoState<MapTile, Entity>,
    pub level_final: Option<TaggedSokoState<MapTile, Entity>>,
    // Maps the boxes where they appear in the initial level to where they appear in the final (solved) state
    pub box_mapping: Option<BoxMapping>,
}

impl Eq for GenState {}

impl Stringable<MapTile, Entity, SokoManager<MapTile, Entity>> for GenState {
    
    fn from_str(s: &String, mgr: &SokoManager<MapTile, Entity>) -> Self {
        let state = <SokoState<MapTile, Entity> as Stringable<MapTile, Entity, SokoManager<MapTile, Entity>>>::from_str(s, mgr);
        return GenState { stage: GenStage::LevelGen,
            level_init: state,
            level_final: None,
            box_mapping: None,
        };
    }

    fn to_str(&self, mgr: &SokoManager<MapTile, Entity>) -> String {
        let stage_str = match self.stage {
            GenStage::LevelGen => { "level_gen" },
            GenStage::SolGen => { "sol_gen" },
            GenStage::Eval => { "eval" }

        };
        //let init_str = SokoState::<MapTile, Entity>::Stringable::to_str(s, mgr);
        let init_str = self.level_init.to_str(mgr);
        match &self.level_final {
            Some(l) => {
                //final_str = SokoState::<MapTile, Entity>::Stringable::to_str(l, mgr);
                return format!("{}\n{}\n{}", stage_str, init_str, l.to_str(mgr));
            },
            None => {
                return format!("{}\n{}", stage_str, init_str);
            }
        }
    }
}

impl GenState {

    pub fn new(height: usize, width: usize) -> Self {
        let mut locs = Vec::<Vector2<isize>>::new();
        let mut tiles = Array::<MapTile, Ix2>::default((height, width));
        let mut entities = Array::<Entity, Ix2>::default((height, width));
        let yy = height / 2;
        let xx = width / 2;
        let player_pos = Vector2::new(yy as isize, xx as isize);
        //TODO: is x,y the right indexing scheme?
        for x in 0..width {
            for y in 0..height {
                let v = Vector2::new(x as isize, y as isize);
                if v != player_pos {
                    tiles[(x,y)] = MapTile::Wall;
                    entities[(x,y)] = Entity::Blank;
                } else {
                    tiles[(x,y)] = MapTile::Blank;
                    entities[(x,y)] = Entity::Player;
                }
            }
        }
        let init_state = SokoState { map_layer: tiles, entity_layer: entities, player_locs: vec![player_pos] };
        return GenState { stage: GenStage::LevelGen,
            level_init: init_state,
            level_final: None,
            box_mapping: None, };
    }

    fn get_init_box_mapping(&self) -> BoxMapping {
        let mut box_mapping = Vec::new();
        for (i, _) in self.level_init.map_layer.indexed_iter() {
            let v = Vector2::new(i.0 as isize, i.1 as isize);
            if self.level_init.get_entity(&v) == Entity::Block {
                // (initial, current)
                box_mapping.push((v, v));
            }
        }
        return box_mapping;
    }

    pub fn neighbors_levelgen<T: HasVecs>(&self, mgr: &T) -> Vec<(GenAction, Self)> {
        // Find the walls that are next to an empty space -- these can be removed
        let mut removable = Vec::new();
        // Find the blank tiles -- these can be turned into boxes
        let mut blanks = Vec::new();
        for (i, _) in self.level_init.map_layer.indexed_iter() {
            let v = Vector2::new(i.0 as isize, i.1 as isize);
            if self.level_init.get_tile(&v) == MapTile::Wall {
                for d in Direction::iter() {
                    let dv = mgr.d_to_v(d);
                    if self.level_init.get_tile(&(v + dv)) == MapTile::Blank {
                        removable.push(v);
                        // It only needs one open neighbor
                        break;
                    }
                }
            }
            if self.level_init.get_tile(&v) == MapTile::Blank &&
                self.level_init.get_entity(&v) == Entity::Blank {
                blanks.push(v);
            }
        }
        let mut neighbors = Vec::new();
        // 1. Delete Obstacle
        for remove in removable {
            let mut neighbor = self.clone();
            neighbor.level_init.set_tile(&remove, MapTile::Blank);
            neighbors.push((GenAction::DelObstacle, neighbor));
        }
        // 2. Place Box
        for blank in blanks {
            let mut neighbor = self.clone();
            neighbor.level_init.set_entity(&blank, Entity::Block);
            neighbors.push((GenAction::AddBox, neighbor));
        }
        // 3. Freeze Level
        let n3 = GenState { stage: GenStage::SolGen,
            level_init: self.level_init.clone(),
            level_final: Some((self.level_init.clone(), 0)),
            box_mapping: Some(self.get_init_box_mapping()) };
        neighbors.push((GenAction::Freeze, n3));
        return neighbors;
    }

    fn get_level_final(&self) -> &SokoState<MapTile, Entity> {
        return &self.level_final.as_ref().expect("No level final!").0;
    }

    fn update_box_mapping<T: HasVecs>(&self, d: Direction, n: &SokoState<MapTile, Entity>, mgr: &T) -> BoxMapping {
        let dv = mgr.d_to_v(d);
        let level_final = self.get_level_final();
        let mut blocks_moved: Vec<Vector2<isize>> = Vec::new();
        // This makes use of the fact that plocs will be the same over time
        // That is, each unique index of plocs corresponds to a unique player (though obviously most of the time there is only one)
        for (ploc_a, ploc_b) in zip(&level_final.player_locs, &n.player_locs) {
            // The proposed move succeeded
            if ploc_a != ploc_b {
                // If ploc_b is the old location of a box, then that box must have moved in direction d
                if level_final.get_entity(&ploc_b) == Entity::Block {
                    //println!("{}, {}", ploc_a, ploc_b);
                    blocks_moved.push(*ploc_b);
                    // Make sure it actually moved
                    assert!(n.get_entity(&(ploc_b + dv)) == Entity::Block);
                }
            }
        }
        let mut new_box_mapping = Vec::new();
        for (i, (b0, bf)) in self.box_mapping.as_ref().expect("No box mapping!").iter().enumerate() {
            if blocks_moved.contains(bf) {
                let bff = *bf + dv;
                new_box_mapping.push((*b0, bff));
            } else {
                new_box_mapping.push((*b0, *bf));
            }
        }
        return new_box_mapping;
    }

    pub fn mk_eval<T: HasVecs>(&self, mgr: &T) -> Self {
        //TODO: a box may have been pushed (and may even NEED to have been pushed) despite ending in the same place.
        // Using box_mapping to determine this info is unreliable
        // All boxes never pushed -> obstacles
        let mut new_box_mapping = Vec::new();
        let mut level_init = self.level_init.clone();
        let mut level_final = self.get_level_final().clone();
        for (i, f) in self.box_mapping.as_ref().expect("No box mapping!") {
            let mut changed = false;
            if i == f {
                level_init.set_tile(&i, MapTile::Wall);
                level_init.set_entity(&i, Entity::Blank);
                level_final.set_tile(&i, MapTile::Wall);
                level_final.set_entity(&i, Entity::Blank);
                changed = true;
            } else {
                // Boxes pushed once -> empty spaces
                for d in Direction::iter() {
                    let dv = mgr.d_to_v(d);
                    let ii = i + dv;
                    if ii == *f {
                        level_init.set_entity(&i, Entity::Blank);
                        level_final.set_entity(&f, Entity::Blank);
                        changed = true;
                        break;
                    }
                }
            }
            // If it wasn't changed, add it to the new box mapping and create a corresponding goal
            if !changed {
                new_box_mapping.push((*i, *f));
                level_init.set_tile(&f, MapTile::Target);
                level_final.set_tile(&f, MapTile::Target);
            }
        }
        // Make sure the generated level is actually winning
        assert!(level_final.is_win());
        return GenState { stage: GenStage::Eval,
            level_init: level_init,
            level_final: Some((level_final, self.level_final.as_ref().expect("No level_final!").1 + 1)),
            box_mapping: Some(new_box_mapping),
        };
    }

    pub fn neighbors_solgen(&self, mgr: &SokoManager<MapTile, Entity>) -> Vec<(GenAction, Self)> {
        let mut neighbors = Vec::new();
        for (a, n) in self.level_final.as_ref().expect("No final level during solgen!").neighbors(mgr) {
            let new_mapping = self.update_box_mapping(a, &n.0, mgr);
            let neighbor = GenState { stage: GenStage::SolGen,
                level_init: self.level_init.clone(),
                level_final: Some(n),
                box_mapping: Some(new_mapping) };
            neighbors.push((GenAction::MovePlayer, neighbor));
        }
        let eval_level = self.mk_eval(mgr);
        neighbors.push((GenAction::Evaluate, eval_level));
        return neighbors;
    }

}

impl Searchable for GenState {
    type V = SokoManager<MapTile, Entity>;
    type A = GenAction;


    fn neighbors(&self, mgr: &Self::V) -> Vec<(Self::A, Self)> {
        match self.stage {
            GenStage::LevelGen => { return self.neighbors_levelgen(mgr); },
            GenStage::SolGen => { return self.neighbors_solgen(mgr); },
            GenStage::Eval => { return Vec::new() },
        }
    }

    //TODO: could also be terminal if the SolGen state is softlocked
    fn terminal(&self) -> bool {
        return self.stage == GenStage::Eval;
    }

    fn is_win(&self) -> bool {
        return false;
    }
}

//TODO: would be better to put these inside the impl for GenState so that we don't have to make all the fields pub
// Implements the heuristics from the paper
// Congestion v2
// Box Count
// 3x3 Block Count
// The value function reported in the paper is (10 * box count + 5 * congestion v2 + 1 * 3x3 Block Count) / 50
// But they're using UCB-V for the value function, so I'm not sure what the right tuning is... once again


fn count_m_e(state: &SokoState<MapTile, Entity>, m_find: MapTile, e_find: Entity) -> (usize, usize) {
    let mut m_count = 0;
    let mut e_count = 0;
    for (i, _) in state.map_layer.indexed_iter() {
        let v = Vector2::new(i.0 as isize, i.1 as isize);
        let m = state.get_tile(&v);
        let e = state.get_entity(&v);
        if m == m_find {
            m_count += 1;
        }
        if e == e_find {
            e_count += 1;
        }
    }
    return (m_count, e_count);
}

fn count_boxes(state: &GenState) -> usize {
    assert!(state.stage == GenStage::Eval);
    let box_mapping = &state.box_mapping.as_ref().expect("No box mapping!");
    return box_mapping.len();
}

fn count_three_by_three(state: &SokoState<MapTile, Entity>) -> usize {
    let mut n_three_by_threes = 0;
    //TODO: this is obviously a bit inefficient
    for (i, _) in state.map_layer.indexed_iter() {
        let v = Vector2::new(i.0 as isize, i.1 as isize);
        let vv = Vector2::new(i.0 as isize + 3, i.1 as isize + 3);
        let (n_walls, _) = count_within(state, v, vv, MapTile::Wall, Entity::Blank);
        let (n_blank, _) = count_within(state, v, vv, MapTile::Blank, Entity::Blank);
        if n_walls == 9 || n_blank == 9 {
            n_three_by_threes += 1;
        }
    }
    return n_three_by_threes;
}

fn count_within(state: &SokoState<MapTile, Entity>, rect_start: Vector2<isize>, rect_end: Vector2<isize>,
    m_find: MapTile, e_find: Entity) -> (usize, usize) {
    let mut m_count = 0;
    let mut e_count = 0;
    // Includes the end
    for x in rect_start.x..(rect_end.x + 1) {
        for y in rect_start.y..(rect_end.y + 1) {
            let v = Vector2::new(x, y);
            let m = state.get_tile_maybe(&v);
            let e = state.get_entity_maybe(&v);
            if m == Some(m_find) {
                m_count += 1;
            }
            if e == Some(e_find) {
                e_count += 1;
            }
        }
    }
    return (m_count, e_count);
}

fn congestion_v2(state: &GenState) -> OrderedFloat<f64> {
    assert!(state.stage == GenStage::Eval);
    let box_mapping = state.box_mapping.as_ref().expect("No box mapping!");
    let n_boxes = box_mapping.len();
    // Not sure why the paper has (alpha b_i + beta g_i) when b_i == g_i
    // The formula from the paper has to be wrong.
    // b_i isn't a value that depends on i (and neither is g_i), while A_i and o_i are dependent on which box
    // I'm interpreting it as (B + G) / sum(A_i - o_i)
    let n_goals = n_boxes;
    let mut denom = OrderedFloat(0.01);
    for (box_start, box_end) in box_mapping {
        let min_x = std::cmp::min(box_start.x, box_end.x);
        let min_y = std::cmp::min(box_start.y, box_end.y);
        let max_x = std::cmp::max(box_start.x, box_end.x);
        let max_y = std::cmp::max(box_start.y, box_end.y);
        // The area includes both points (e.g. if both points are the same, the area is 1, not 0)
        let A_i = (max_x + 1 - min_x) * (max_y + 1 - min_y);
        let v_start = Vector2::new(min_x, min_y);
        let v_end = Vector2::new(max_x, max_y);
        let (o_i, _) = count_within(&state.level_init, v_start, v_end, MapTile::Wall, Entity::Blank);
        let fill_factor = A_i - o_i as isize;
        denom += OrderedFloat(fill_factor as f64);
    }
    let numer = OrderedFloat(n_goals as f64 + n_boxes as f64);
    return numer / denom;
}

fn eval_end_state(state: &GenState) -> OrderedFloat<f64> {
    let box_count = OrderedFloat(count_boxes(state) as f64);
    let three_by_three = OrderedFloat(count_three_by_three(&state.level_init) as f64);
    let congestion = congestion_v2(state);
    // The value function reported in the paper is (10 * box count + 5 * congestion v2 + 1 * 3x3 Block Count) / 50
    //println!("{}, {}, {}", box_count, three_by_three, congestion);
    return (OrderedFloat(10.0) * box_count + OrderedFloat(5.0) * congestion + three_by_three) / OrderedFloat(50.0);
}

pub fn eval_state(state: &GenState) -> OrderedFloat<f64> {
    match state.stage {
        GenStage::Eval => { return eval_end_state(state); },
        _ => { return OrderedFloat(0.0); }
    }
}
