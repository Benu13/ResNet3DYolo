**## Preparing Box Data**

The code for preparing box data is located in `prepare_boxes.ipynb`. To streamline this process with minimal user intervention, a specialized tool has been developed. This code uses series information creted with `create_series_info_data.ipynb`. 

**Processing Steps for Each Series:**

1. **Load Study:** Load the necessary study data.
2. **Initial Quality Check:** Verify the data for any potential issues.
    * Missing conditions
    * Missplaced conditions
    * Missing levels
3. **Coordinate Unification:** Translate coordinates across different series types.
    * Spinal canale coordinates from Sagittal T2 -> Sagittal T1
    * Foraminal coordinates from Sagittal T1 -> Sagittal T2
    * Canale and Foraminal coordinates from unified Sagittal T1 -> Axial T2 
4. **Subsequent Quality Check:** Conduct a second round of checks post-unification.
    * Missplaced conditions
5. **Issue Resolution:** Address any identified problems using the following strategies:

**Common Issues and Solutions:**

1. **Inconsistent Foramina Data:**
   * If within one study two sagittal T1 series exist, one containing left foramina and the other containing right foramina:
     - Translate coordinates from one series to the other.

2. **Misplaced Foramina Coordinates:**
   * If foraminas coordinates are incorrect but one of them align with translated spinal cord coordinates:
     - Automatically reassign the misplaced foramina to the correct spinal level based on the spinal cord coordinates.
     
   ![alt text](/images/foramina_fix.png)

3. **Misplaced Spinal Canal Coordinates:**
   * If spinal canal coordinates are incorrectly positioned after translation from T2 to T1:
     - **Manual Correction:** Allow the user to manually correct the position of first spinal canal coordinate. The remaining coordinates will be automatically adjusted based on this correction.
     - **Automatic Correction:** If foramina are not misplaced (i.e., not issue #2), automatically position the spinal canal between the foramina.
     - **Combined Correction:** If both issues 2 and 3 occur, first apply the manual correction for issue 3, then automatically apply the correction for issue 2.
