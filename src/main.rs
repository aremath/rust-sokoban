use std::fs;
use ndarray::{Array, array, Ix2};
use text_io::read;

use crate::sokoengine::Stringable;

mod sokoengine;
mod sokoset;

enum Command {
    Quit,
    Undo,
    Direction { d: sokoengine::Direction },
    Nothing,
}

fn char_to_command(c: Option<&u8>) -> Command {
        match c {
            // wasd controls
            Some(119) => Command::Direction { d: sokoengine::Direction::Up },
            Some(97) => Command::Direction { d: sokoengine::Direction::Left },
            Some(115) => Command::Direction { d: sokoengine::Direction::Down },
            Some(100) => Command::Direction { d: sokoengine::Direction::Right },
            // x to quit
            Some(120) => Command::Quit,
            // u to undo
            Some(117) => Command::Undo,
            // Any other input should do nothing, but not quit
            Some(_) => Command::Nothing,
            None => Command::Nothing,
        }
}

fn main() {
    /*
    let manager: sokoengine::SokoManager<sokoengine::MapTile, sokoengine::Entity>
        = sokoengine::SokoManager::new(sokoengine::mk_type_to_text);
    */
    let submanager: sokoengine::SokoManager<sokoset::MapSet, sokoset::EntitySet> =
        sokoengine::SokoManager::new(sokoset::mk_set_to_text);
    let patterns = sokoset::Patterns::new();
    let manager = sokoset::SetManager::new(submanager, patterns);
    let contents = fs::read_to_string("./src/sokoban_1_t.lvl").expect("Couldn't read the file!");
    //let contents = fs::read_to_string("./src/sokoban_multiple_players.lvl").expect("Couldn't read the file!");
    // Create the memory object that holds all the states
    /*
    let mut s_mem: sokoengine::SokoMemory<sokoengine::MapTile, sokoengine::Entity, sokoengine::SokoManager<sokoengine::MapTile, sokoengine::Entity>>
        = sokoengine::SokoMemory::from_str(&contents, &manager);
    */
    let mut s_mem: sokoengine::SokoMemory<sokoset::MapSet, sokoset::EntitySet, sokoset::SetManager> =
        sokoengine::SokoMemory::from_str(&contents, &manager);
    println!("LEVEL: \n{}", s_mem.to_str(&manager));
    loop {
        // Get the text input
        let i: String = read!();
        let c: Option<&u8> = i.as_bytes().get(0);
        //println!("{:?}", c);
        let command = char_to_command(c);
        match command {
            Command::Quit => {
                    println!("YOU QUIT");
                    break;
                }
            Command::Direction { d: direction} => {
                println!("{:?}", direction);
                s_mem.update(direction, &manager);
            }
            Command::Undo => {
                s_mem.undo();
            }
            Command::Nothing => {}
        }
        //println!("{}", (*s_box).to_str(&manager));
        println!("{}", s_mem.to_str(&manager));
        if s_mem.is_win() {
            println!("YOU WIN!");
            break;
        }
    }
}
