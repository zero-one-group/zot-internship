#!/bin/bash

dotfiles=(".gitconfig" ".zshrc" ".vimrc")
dir="${HOME}/zot-internship/zot-internship/maxine/dotfiles"

for dotfile in "${dotfiles[@]}";
do 
	ln -sf "${HOME}${dotfile}" "${dir}"
done

