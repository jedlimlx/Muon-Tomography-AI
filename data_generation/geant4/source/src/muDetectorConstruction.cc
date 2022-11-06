
#include "muDetectorConstruction.hh"
#include "muSensorSD.hh"
#include "muAnalyzer.hh"

#include "G4Material.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Orb.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4UniformMagField.hh"
#include "G4SDManager.hh"
#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4UserLimits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4RunManager.hh"
#include "G4NistManager.hh" // [yy] for use of G4materials

#include "PerlinNoise.hh"

#include <iostream>
#include <string>

muDetectorConstruction::muDetectorConstruction()
:solidSensor(0),logicSensor(0),physSensor(0){

    DefineMaterials();
}

muDetectorConstruction::~muDetectorConstruction()
{

}

//------------------------------------------------------------------------//
// Begin of Construct()
G4VPhysicalVolume* muDetectorConstruction::Construct()
{
    // Clean old geometry, if any
    G4GeometryManager::GetInstance()->OpenGeometry();
    G4PhysicalVolumeStore::GetInstance()->Clean();
    G4LogicalVolumeStore::GetInstance()->Clean();
    G4SolidStore::GetInstance()->Clean();

    //------------------------------------------------
    // Create geometry
    //------------------------------------------------

   // --- Overview of geometry ---
   // World - Module - Scabox  - Sensor* (copy no.: 0-63)
   //                             Absbox  - Sensor* (copy no: 64-127)
   // * means "Sensitive Detector"

   // --- Parameters for geometry ---

   // World
    G4double world_sizeX = 100*cm;
    G4double world_sizeY = 100*cm;
    G4double world_sizeZ = 100*cm;

  // Scabox/Absbox
  G4double box_sizeX =     30.0  *mm;  // > voxel_sizeX
  G4double box_sizeY =     24.0  *mm;  // > voxel_pitchY*number_of_voxel_dimY
  G4double box_sizeZ =     24.0  *mm;  // > voxel_pitchZ*number_of_voxel_dimZ

  G4double cont_sizeX =     10.0  *mm;  // > voxel_sizeX
  G4double cont_sizeY =     10.0  *mm;  // > voxel_pitchY*number_of_voxel_dimY
  G4double cont_sizeZ =     10.0  *mm;  // > voxel_pitchZ*number_of_voxel_dimZ


  // Sensor
  G4double voxel_sizeX =   30.0  *mm;
  G4double voxel_sizeY =     2.0  *mm;
  G4double voxel_sizeZ =     2.0  *mm;
  G4double voxel_pitchY =    3.0  *mm;
  G4double voxel_pitchZ =    3.0  *mm;
  G4int number_of_voxel_dimY  =   8;
  G4int number_of_voxel_dimZ  =   8;

    // distances
    G4double distance_source_sca  = 100.0  *mm; // distance between scabox and origin(0,0,0)
    G4double distance_sca_abs        =  80.0  *mm; // distance between absorber and scatter

    // module
    G4double module_sizeX = box_sizeX+distance_sca_abs+box_sizeX;
    G4double module_sizeY = box_sizeY; // under supposition that scabox and absbos are same
    G4double module_sizeZ = box_sizeZ; // under supposition that scabox and absbos are same

    // if create multiple modules as circle
    //G4int number_of_modules = 1;  // 1: single Compton camera module
    //G4double angle_rot_modules = 360.0/number_of_modules *deg;

    // -- World
    // solid definition (size)
    G4Box* solidWorld =  new G4Box("World",                     // its name
	                                                       0.5*world_sizeX, 0.5*world_sizeY, 0.5*world_sizeZ);      // its size

    // logical volume definition (material)
    G4LogicalVolume* logicWorld =
    new G4LogicalVolume(
                        solidWorld,          // its solid
                        Air,                 // its material
                        "World");            // its name

  // physical volume definition (position, relation, copy number)
  G4VPhysicalVolume* physWorld =
    new G4PVPlacement(
                      0,                                // no rotation
                      G4ThreeVector(),       // at (0,0,0)
                      logicWorld,          // its logical volume
                      "World",               // its name
                      0,                         // its mother  volume
                      false,                   // no boolean operation
                      0,                         // copy number
                      false);       // checking overlaps

  // --- Module

    // solid definition (size)
    G4Box* solidModule =
    new G4Box("Module",                       // its name
	      0.5*module_sizeX, 0.5*module_sizeY, 0.5*module_sizeZ); // its size

    // logical volume definition (material)
    G4LogicalVolume* logicModule =
    new G4LogicalVolume(solidModule,          // its solid
                        Air,                  // its material
                        "Module");            // its name

    // physical volume definition (position, relation, copy number)
    new G4PVPlacement(0,         // no rotation
                      G4ThreeVector((distance_source_sca+module_sizeX*0.5),0,0),
                      logicModule,          // its logical volume
                      "Module",               // its name
                      logicWorld,            // its mother  volume
                      false,                 // no boolean operation
                      0,                       // copy number
                      false);                // checking overlaps

    /* // if create multiple modules as circle
    for (G4int i = 0; i < number_of_modules ; i++) {

      G4ThreeVector armVec= G4ThreeVector((distance_source_sca+module_sizeX*0.5),0,0);
      G4RotationMatrix armRot= G4RotationMatrix();
      armVec.rotateZ(i*angle_rot_modules);
      armRot.rotateZ(i*angle_rot_modules);

      new G4PVPlacement(G4Transform3D(armRot, armVec),
                      logicModule,          // its logical volume
                      "Module",               // its name
                      logicWorld,            // its mother  volume
                      false,                 // no boolean operation
                      i,                       // copy number
                      true);                // checking overlaps
     }
   */




  // -- scatter box

  // solid definition (size)
  G4Box* solidScabox =
    new G4Box("Scabox",                       // its name
	      0.5*box_sizeX, 0.5*box_sizeY, 0.5*box_sizeZ); // its size

  // logical volume definition (material)
  G4LogicalVolume* logicScabox =
    new G4LogicalVolume(solidScabox,          // its solid
                        Air,              // its material
                        "Scabox");            // its name

  new G4PVPlacement(0,                        // no rotation
                      G4ThreeVector(box_sizeX*0.5 - module_sizeX*0.5, 0, 0),
                      logicScabox,            // its logical volume
                      "Scabox",               // its name
                      logicModule,            // its mother  volume
                      false,                  // no boolean operation
                      0,                      // copy number
                      true);        // checking overlaps

  // -- absober box

  // solid definition (size)
  G4Box* solidAbsbox =
    new G4Box("Absbox",                       // its name
	      0.5*box_sizeX, 0.5*box_sizeY, 0.5*box_sizeZ); // its size

  // logical volume definition (material)
   G4LogicalVolume* logicAbsbox =
    new G4LogicalVolume(solidAbsbox,          // its solid
                        Air,                  // its material
                        "Absbox");            // its name

  // physical volume definition (position, relation, copy number)
    new G4PVPlacement(0,                      // no rotation
                      G4ThreeVector(module_sizeX*0.5 - box_sizeX*0.5, 0, 0),
                      logicAbsbox,            // its logical volume
                      "Absbox",               // its name
                      logicModule,             // its mother  volume
                      false,                  // no boolean operation
                      1,                      // copy number
                      true);        // checking overlaps


    // Control volume
    // solid definition (size)
    /*
  G4Box* solidContBox =
      new G4Box("Contbox",
                0.5*cont_sizeX,0.5*cont_sizeY,0.5*cont_sizeZ);

  // logical volume definition (material)
  G4LogicalVolume* logicContbox =
    new G4LogicalVolume(solidContBox,          // its solid
                        Lead,              // its material
                        "Contbox");            // its name

  new G4PVPlacement(0,                        // no rotation
                      G4ThreeVector(0, 0, 0),
                      logicContbox,            // its logical volume
                      "Contbox",               // its name
                      logicModule,            // its mother  volume
                      false,                  // no boolean operation
                      2,                      // copy number
                      true);        // checking overlaps
    */

  PerlinNoise pn(time(NULL));
  double v;
    const int RESOLUTION = 64;

                // solid definition (size)
  G4Box* solidContBox =
    new G4Box("Contbox",
              cont_sizeX / (2 * RESOLUTION) * mm, cont_sizeY / (2 * RESOLUTION) * mm , cont_sizeZ / (2 * RESOLUTION) * mm);
    // logical volume definition (material)
  G4LogicalVolume* logicContbox =
      new G4LogicalVolume(solidContBox,          // its solid
                          Lead,              // its material
                          "Contbox");            // its name

  std::ifstream infile;
  infile.open(filename + ".voxel");

  if (infile.is_open()){
    double x,y,z;
    x=0;y=0;z=0;
    for (int i = 0; i < RESOLUTION; i++){
        for (int j = 0; j < RESOLUTION; j++){
            for (int k = 0; k < RESOLUTION; k++){
              x = i / (double) RESOLUTION;
              y = j / (double) RESOLUTION;
              z = k / (double) RESOLUTION;
                infile >> v;
                if (v) {
                    new G4PVPlacement(0,                        // no rotation
                                      G4ThreeVector((x-0.5) * cont_sizeX, (y-0.5) * cont_sizeY, (z-0.5) * cont_sizeZ),
                                      logicContbox,            // its logical volume
                                      "ContboxAdd",               // its name
                                      logicModule,            // its mother  volume
                                      false,                  // no boolean operation
                                      i * RESOLUTION * RESOLUTION + j * RESOLUTION + k,                      // copy number
                                      false);        // not checking overlaps
                } 
            }
        }
        std::cout << i << std::endl;
    }
    
  }else{
    throw std::runtime_error("could not open file");
  }

  infile.close();





  // -- scatter voxel

  // solid definition (size)
  solidSensor =
    new G4Box("Sensor",                       // its name
	                    0.5*box_sizeX, 0.5*box_sizeY, 0.5*box_sizeZ); // its size

  // logical volume definition (material)
  logicSensor =
    new G4LogicalVolume(solidSensor,          // its solid
                        EJ200,                   // its material
                        "Sensor");            // its name
    new G4PVPlacement(0,                    // no rotation
                      G4ThreeVector(0,0,0), // location
                    logicSensor,          // its logical volume
                    "Sensor",             // its name
                    logicAbsbox,            // its mother  volume
                    false,                  // no boolean operation
                    0,  // copy number
                    true);        // checking overlaps

    new G4PVPlacement(0,                    // no rotation
                      G4ThreeVector(0,0,0), // location
                        logicSensor,          // its logical volume
                        "Sensor",             // its name
                        logicAbsbox,            // its mother  volume
                        false,                  // no boolean operation
                        0,  // copy number
                        true);        // checking overlaps


    /*
  // physical volume definition (position, relation, copy number)
  for (G4int i = 0; i < number_of_voxel_dimY ; i++) {
    for (G4int j = 0; j < number_of_voxel_dimZ ; j++) {
       new G4PVPlacement(0,                     // no rotation
                      G4ThreeVector(0,(i+0.5)*voxel_pitchY-box_sizeY*0.5,(j+0.5)*voxel_pitchZ-box_sizeZ*0.5), //location
                      logicSensor,            // its logical volume
                      "Sensor",               // its name
                      logicScabox,              // its mother  volume
                      false,                    // no boolean operation
                      i*number_of_voxel_dimZ+j,      // copy number
                      true);                       // checking overlaps
    }
  }

  // physical volume definition (position, relation, copy number)
  for (G4int i = 0; i < number_of_voxel_dimY ; i++) {
    for (G4int j = 0; j < number_of_voxel_dimZ ; j++) {
      new G4PVPlacement(0,                    // no rotation
                      G4ThreeVector(0,(i+0.5)*voxel_pitchY-box_sizeY*0.5,(j+0.5)*voxel_pitchZ-box_sizeZ*0.5), // location
                      logicSensor,          // its logical volume
                      "Sensor",             // its name
                      logicAbsbox,            // its mother  volume
                      false,                  // no boolean operation
                      i*number_of_voxel_dimZ+j + number_of_voxel_dimY*number_of_voxel_dimZ,  // copy number
                      true);        // checking overlaps
    }
  }
    */

    //------------------------------------------------
    // Sensitive detectors
    //------------------------------------------------

    G4SDManager* SDman = G4SDManager::GetSDMpointer();
    muSensorSD* aSensorSD = (muSensorSD*)SDman->FindSensitiveDetector("mu/SensorSD");
    if ( aSensorSD == 0){
        aSensorSD = new muSensorSD("mu/SensorSD");
        SDman->AddNewDetector( aSensorSD );
    }
    aSensorSD->SetAnalyzer(analyzer);
    logicSensor->SetSensitiveDetector(aSensorSD);

    //------------------------------------------------
    // Visualization attributes
    //------------------------------------------------

    logicWorld->SetVisAttributes (G4VisAttributes::Invisible);

    G4VisAttributes* simpleBoxVisAtt= new G4VisAttributes(G4Colour(0.0,1.0,1.0));
    simpleBoxVisAtt->SetVisibility(true);
    logicSensor->SetVisAttributes(simpleBoxVisAtt);

    return physWorld;
}
// End of Construct()
//------------------------------------------------------------------------//


///////////////////////////////////////////////////////
void muDetectorConstruction::DefineMaterials(){
    /*This function illustrates the possible ways to define materials*/

    // Getting the NIST Material Manager [yy]

	nistMan = G4NistManager::Instance();
	nistMan->SetVerbose(0);

    // Define parameters

    G4String symbol, name;
    G4double a; // mass of a mole
    G4double z; // mean number of protons
    G4double density;

    // Define Elements

    G4Element* O  = new G4Element("Oxygen"  ,symbol="O" , z= 8., a= 16.00*g/mole);
    G4Element* Al = new G4Element("Aluminium",symbol="Al" , z= 13., a= 26.98*g/mole);
    G4Element* Gd = new G4Element("Gadolinium", "Gd", z=64, a=157.25    *g/mole); // [yy]
    G4Element* Ga = new G4Element("Gallium"   , "Ga", z=31, a= 69.723   *g/mole); // [yy]

    // Define materials

	Air = nistMan->FindOrBuildMaterial("G4_AIR");  // [yy] Air
	EJ200 = nistMan->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE"); // [yy] Eljen, EJ200
    Lead = nistMan->FindOrBuildMaterial("G4_Au"); // Actually uranium


	GAGG = new G4Material("GAGG", density=6.63*g/cm3, 4); // [yy] GAGG scintillator (not used now)
	GAGG -> AddElement(Gd,  3);
	GAGG -> AddElement(Al,  2);
	GAGG -> AddElement(Ga,  3);
	GAGG -> AddElement(O , 12);

	G4cout << *(G4Material::GetMaterialTable()) << G4endl;

}

void muDetectorConstruction::UpdateGeometry()
{
    G4RunManager::GetRunManager()->DefineWorldVolume(Construct());
}

void muDetectorConstruction::SetAnalyzer(muAnalyzer * analyzer_in)
{
    analyzer = analyzer_in;
}
void muDetectorConstruction::SetVoxelFileName(TString name){
    filename = name;
}
