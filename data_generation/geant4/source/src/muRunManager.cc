
#ifdef G4MULTITHREADED
#include "G4MTRunManager.hh"
#else
#include "G4RunManager.hh"
#endif

#include "muRunManager.hh"

muRunManager::muRunManager() : G4RunManager()
{
    
}
