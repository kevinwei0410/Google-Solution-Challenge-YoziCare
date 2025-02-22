import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class FoodImage extends StatelessWidget {
  // const FoodImage({
  //   Key ? key
  // }): super(key: key);
  final File food_file;
  FoodImage(this.food_file);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 230. w,
      margin: EdgeInsets.only(top: 20. w),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(25),
      ),
      child: Stack(
        children: [
          ClipRRect(
              borderRadius: BorderRadius.circular(25),
              child: Image.file(food_file, fit: BoxFit.cover, height: 230. w, width: double.infinity,), //Image.asset(food_file, fit: BoxFit.cover, height: 230. w, width: double.infinity, )
          ),
          // Align(
          //   alignment: Alignment.bottomLeft,
          //   child: Container(
          //       margin: EdgeInsets.only(left: 15. w, bottom: 20. w, ),
          //       child: Text(
          //         'Pasta with tomato sauce',
          //         style: TextStyle(
          //             color: Colors.white,
          //             fontSize: 22. sp
          //         ),
          //       )
          //   ),
          // )
        ],
      ),
    );
  }
}