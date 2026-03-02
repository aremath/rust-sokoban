extern crate nalgebra as na;
use na::Vector2;
use bimap::BiMap;
use ndarray::{Array, Ix2};
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::sokoengine;

const NUM_PATTERNS: usize = 2;

//TODO: some way to use a single u16 instead of tuples for (map, entity)?

pub type MapSet = u8;
pub type EntitySet = u8;

const MAP_BLANK: MapSet = 1;
const MAP_WALL: MapSet = 2;
const MAP_TARGET: MapSet = 4;
const MAP_ANY: MapSet = 7;
const MAP_NOWALL: MapSet = 5;

const ENTITY_BLANK: EntitySet = 1;
const ENTITY_BLOCK: EntitySet = 2;
const ENTITY_PLAYER: EntitySet = 4;
const ENTITY_ANY: EntitySet = 7;

impl sokoengine::IsPlayer for EntitySet {
    fn is_player(&self) -> bool {
        return *self == ENTITY_PLAYER;
    }
}

//TODO: cannot implement Default for u8, but it should be fine. It's only really used for out of bounds access...

// Create the symbol table for sokoset
pub fn mk_set_to_text() -> BiMap<(MapSet, EntitySet), char> {
    let mut set_to_text = BiMap::new();
    set_to_text.insert((MAP_BLANK, ENTITY_BLANK),    '_');
    set_to_text.insert((MAP_BLANK, ENTITY_BLOCK),    'B');
    set_to_text.insert((MAP_BLANK, ENTITY_PLAYER),   'P');
    set_to_text.insert((MAP_WALL,  ENTITY_BLANK),    'W');
    set_to_text.insert((MAP_TARGET, ENTITY_BLANK),   'O');
    set_to_text.insert((MAP_TARGET, ENTITY_BLOCK),   'G');
    set_to_text.insert((MAP_TARGET, ENTITY_PLAYER),  'Q');
    return set_to_text;
}

pub struct SetManager {
    m: sokoengine::SokoManager<MapSet, EntitySet>,
    patterns: Patterns,
}

impl SetManager {

    pub fn new(m: sokoengine::SokoManager<MapSet, EntitySet>, patterns: Patterns) -> Self {
        return SetManager { m: m, patterns: patterns };
    }

}

impl sokoengine::Coder<MapSet, EntitySet> for SetManager {
    fn encode(&self, t: (MapSet, EntitySet)) -> char {
        match self.m.symbol_table.get_by_left(&t) {
            Some(&c) => c,
            None => '?' // For any tile which has multiple possibilities
        }
    }

    fn decode(&self, c: char) -> (MapSet, EntitySet) {
        return self.m.decode(c);
    }
}

impl sokoengine::HasVecs for SetManager {

    fn d_to_v(&self, d: sokoengine::Direction) -> &Vector2<isize> {
        return self.m.d_to_v(d);
    }

}

#[derive(Clone)]
pub struct PatternEntry {
    cond: (MapSet, EntitySet), // Preconditions that are required to apply the pattern
    post: (MapSet, EntitySet)  // Things that happen as a result of applying the pattern (0 if nothing)
}

pub struct Patterns {
    basic_move: [PatternEntry; 2],
    basic_push: [PatternEntry; 3],
}

/*
TODO: v how to do this? v
impl IntoIterator for Patterns {
    type Item = &[PatternEntry];
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(&self) -> Self::IntoIter {
        let v = vec![&self.basic_move, &self.basic_push];
        return v.iter();
    }
}
*/

impl Patterns {

   pub fn new() -> Self {
        let basic_move_a = PatternEntry { cond: (MAP_NOWALL, ENTITY_PLAYER), post: (0, ENTITY_BLANK) };
        let basic_move_b = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLANK), post: (0, ENTITY_PLAYER) };
        let basic_move = [basic_move_a, basic_move_b];
        //
        let basic_push_a = PatternEntry { cond: (MAP_NOWALL, ENTITY_PLAYER), post: (0, ENTITY_BLANK) };
        let basic_push_b = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLOCK), post: (0, ENTITY_PLAYER) };
        let basic_push_c = PatternEntry { cond: (MAP_NOWALL, ENTITY_BLANK), post: (0, ENTITY_BLOCK) };
        let basic_push = [basic_push_a, basic_push_b, basic_push_c];
        return Patterns { 
            basic_move: basic_move,
            basic_push: basic_push,
        };
    }

}

type SokoSet = sokoengine::SokoState<MapSet, EntitySet>;

impl SokoSet {
    
    // Updates self with the given pattern, returning whether the pattern succeeded
    // This allows multiple patterns to be applied without a clone()
    fn mut_pattern_at<V: sokoengine::HasVecs>(&mut self, c: &Vector2<isize>, d: sokoengine::Direction,
        pattern: &[PatternEntry], mgr: &V) -> bool {
        let mut current_pos = c.clone();
        let dv = mgr.d_to_v(d);
        // Check if the preconditions are met
        for p in pattern {
            let m = self.get_tile(&current_pos);
            let e = self.get_entity(&current_pos);
            // The rule fails if either possibility set is empty
            if ((m & p.cond.0) == 0 || (e & p.cond.1) == 0) {
                return false;
            }
            current_pos = current_pos + dv;
        }
        // If the preconditions are met, then apply them.
        // Also apply any necessary postconditions
        current_pos = c.clone();
        for p in pattern {
            if p.post.0 != 0 {
                self.set_tile(&current_pos, p.post.0);
            } else {
                let m = self.get_tile(&current_pos);
                self.set_tile(&current_pos, m & p.cond.0);
            }
            if p.post.1 != 0 {
                self.set_entity(&current_pos, p.post.1);
            } else {
                let e = self.get_entity(&current_pos);
                self.set_entity(&current_pos, e & p.cond.1);
            }
            current_pos = current_pos + dv;
        }
        return true;
    }

    // Try to apply a pattern at a specific location. Other functions will determine where to apply them
    fn apply_pattern_at<V: sokoengine::HasVecs>(&self, c: &Vector2<isize>, d: sokoengine::Direction,
        pattern: &[PatternEntry], mgr: &V) -> Option<SokoSet> {
        let mut new_state = self.clone();
        let mut current_pos = c.clone();
        let dv = mgr.d_to_v(d);
        // Check if the preconditions are met
        for p in pattern {
            let m = self.get_tile(&current_pos);
            let e = self.get_entity(&current_pos);
            if ((m & p.cond.0) == 0 || (e & p.cond.1) == 0) {
                return None;
            }
            current_pos = current_pos + dv;
        }
        // If the preconditions are met, apply the postconditions
        current_pos = c.clone();
        for p in pattern {
            if p.post.0 != 0 {
                new_state.set_tile(&current_pos, p.post.0);
            } else {
                let m = new_state.get_tile(&current_pos);
                new_state.set_tile(&current_pos, m & p.cond.0);
            }
            if p.post.1 != 0 {
                new_state.set_entity(&current_pos, p.post.1);
            } else {
                let e = new_state.get_entity(&current_pos);
                new_state.set_entity(&current_pos, e & p.cond.1);
            }
            current_pos = current_pos + dv;
        }
        return Some(new_state);
    }

    //TODO: do I need a version of this that applies multiple patterns mutably?
    fn apply_at_plocs<V: sokoengine::HasVecs>(&self, d: sokoengine::Direction,
        patterns: &[&[PatternEntry]], mgr: &V) -> Option<SokoSet> {
        //println!("OLD {:?}", self.player_locs);
        let ordering = sokoengine::choose_ordering(d);
        let dv = mgr.d_to_v(d);
        let mut locs = self.player_locs.to_vec();
        let mut new_state = self.clone();
        let mut changed = false;
        let mut new_plocs: Vec<Vector2<isize>> = Vec::new();
        locs.sort_by_key(ordering);
        for loc in locs {
            let mut loc_changed = false;
            for pattern in patterns {
                let b = new_state.mut_pattern_at(&loc, d, pattern, mgr);
                // If any pattern succeeds, then the state is changed*
                changed = changed || b;
                loc_changed = loc_changed || b;
                if b {
                    //TODO: Not all patterns will actually change the player position when they succeed
                    let loc_next = loc + dv;
                    new_plocs.push(loc_next);
                    // At most one pattern is applied per step
                    break;
                }
            }
            // If no patterns were applied, we still need to retain the location
            if !loc_changed {
                //println!("Not Changed!");
                new_plocs.push(loc);
            }
        }
        if changed {
            //println!("NEW {:?}", new_plocs);
            new_state.player_locs = new_plocs;
            return Some(new_state);
        } else {
            return None;
        }
    }
}

impl sokoengine::SokoInterface<MapSet, EntitySet, SetManager> for SokoSet {

    fn update(&self, d: sokoengine::Direction, mgr: &SetManager) -> Option<Self> {
        // Create slices for each applicable pattern
        let pa: &[PatternEntry] = &mgr.patterns.basic_move;
        let pb: &[PatternEntry] = &mgr.patterns.basic_push;
        let patterns: &[&[PatternEntry]] = &[pa, pb];
        return self.apply_at_plocs(d, patterns, mgr);
    }

    //TODO: could consider some definitions of victory such as:
    //  1. All possible configurations win
    //  2. There exists a winning configuration
    //  Both of these seem problematic
    fn is_win(&self) -> bool {
        for (i, _) in self.map_layer.indexed_iter() {
            if self.map_layer[i] == MAP_TARGET && !(self.entity_layer[i] == ENTITY_BLOCK) {
                return false;
            }
        }
        return true;
    }
}
