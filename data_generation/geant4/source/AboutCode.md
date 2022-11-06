# AboutCode 


## ジオメトリの定義

ジオメトリとは、検出器やその他の周辺物体を物質や大きさ、配置などを定義している。

空気環境下において、EJ-200 のシンチレータ が 8x8 で配列されたアレイが、X軸方向に二つ並べて配置されている（下図）。


それぞれのアレイを**Scabox**と**Absbox**と定義しており、これらは1つのモジュール (**Module**)の中に入っている。

 Scabox と Abscox は、それぞれ 8x8 の EJ-200 シンチレータ (**voxel**) から構成されていて、**voxel**の大きさは、2x2x30mmである。
**voxel** は “sensitive detector”. にアサインされている。

ジオメトリは、**source/src/muDetectorConstruction.cc (define geometry)** に定義されているので、ここを編集する。

### 補足1) voxel の copy number 

どのシンチに当たったかを特定するために、128 個 (8x8x2) のシンチには copy number (copyNo) が割り当てられている。  

 - scabox の voxel: 0-63
 - absbox の voxel: 64-127

データ解析の時に必要になる。

### 補足2) EJ-200/EJ-212 の定義


EJ-200/EJ-212 は、ここでは ビニルトルエン として定義している。  
(参考: [murffer/DetectorSim/blob/master/ScintillationSlab/src/Material.cc](https://github.com/murffer/DetectorSim/blob/master/ScintillationSlab/src/Material.cc))  

```muDetectorConstruction.cc
EJ200 = nistMan->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
```

ちなみに、このビニルトルエン (G4\_PLASTIC\_SC\_VINYLTOLUENE) は  G4NistMaterialBuilder.cc の 1468-1470 行に定義してある。  
(参考: [Id: G4NistMaterialBuilder.cc 67044 2013-01-30 08:50:06Z gcosmo](http://www.apc.univ-paris7.fr/~franco/g4doxy/html/G4NistMaterialBuilder_8cc-source.html))

```G4NistMaterialBuilder.cc
AddMaterial("G4_PLASTIC_SC_VINYLTOLUENE", 1.032, 0, 64.7, 2);
AddElementByWeightFraction( 1, 0.085);
AddElementByWeightFraction( 6, 0.915);
```


## ソース (線源) の定義

 
 ソース (線源) は、**GPS (General Particle Source)** を用いて定義する。  
 GPS の設定は、**bench/run.mac** (マクロファイル) で行う。
 
 GPS の参考になる資料は以下の二つ。
 
 - **[Using GPS (The General Particle Source)](http://nngroup.physics.sunysb.edu/captain/reference/master/detSim/dox/detSimGPS.html)**  
- **[Geant4 User Guide](ftp://ftp.iij.ad.jp/pub/linux/gentoo/distfiles/BookForAppliDev-4.10.2.pdf)**

