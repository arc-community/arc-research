# Overview

## Icecuber DSL function types


1. Unary:  
 `Image` -> `Image`
2. Binary:  
 `(Image, Image)` -> `Image`
3. Split:  
 `Image` -> `List[Image]`
4. Join:   
 `List[Image]` -> `Image`
5. Vector:  
 `List[Image]` -> `List[Image]`

## Func Examples

## binary.ipynb
- "embed"
- "wrap"
- "broadcast"
- "repeat_0"
- "repeat_1"
- "mirror_0"
- "mirror_1"
- "filter_color_palette"
- "replace_colors"

## border_connect.ipynb
- "border"
- "make_border"
- "make_border2_{b}"
- "connect_{id}"

## coloring.ipynb
- "filter_color_{i}"
- "erase_color_{i}"
- "color_shape_const_{i}"
- "majority_color_image"
- "spread_colors_{1 if skipmaj else 0}"

## compress.ipynb
- "compress"
- "compress2"
- "compress3"

## join.ipynb
- "pick_unique"
- "compose_growing"
- "stack_line"
- "my_stack_list_{id}"
- "pick_max_{id}"
    
## move.ipynb
- "move_{dx}_{dy}" 

## other_unary.ipynb
- "get_pos"
- "get_size"
- "get_size0"
- "hull"
- "hull0"
- "to_origin"
- "fill"
- "interior"
- "interior2"
- "center"

## pick_maxes.ipynb
- "pick_maxes_{id}"
- "pick_not_maxes_{id}"

## rigid_half.ipynb
- "rigid_{i}"
- "half_{i}"

## smear.ipynb
- "smear_{i}"

## split.ipynb
- "cut_image"
- "split_colors"
- "split_all"
- "split_columns"
- "split_rows"
- "inside_marked"
- "gravity_{d}"
